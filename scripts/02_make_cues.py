"""Stage 2 - cue maker (CPU): computes the classical-CV cues once for every training and test clip.

Outputs (in --out):
  ego_features.csv   keypoint-geometry features per clip
  ego_model.joblib   motion classifier (stopped / slowing down / speeding up / turning)
  cues_all.json      per clip: motion sentence, traffic-light boxes, vanishing point
  pool_segments.txt  training clips, in training order

python scripts/02_make_cues.py --lingoqa /path/to/lingoqa --out outputs/cues
"""
import argparse, os, json, collections, traceback
from multiprocessing import Pool
import pandas as pd
from perceptqa.data import LingoQA
from perceptqa.cues.geometry import ego_features, gray_small, features, match, pair_geometry, K_of
from perceptqa.cues.lights import light_candidates
from perceptqa.cues.road_ahead import vanishing_point
from perceptqa.cues.ego import EgoEstimator, mine_label

def _features(job):
    d, seg = job
    try: return ego_features(d, seg) or {"segment_id": seg}
    except Exception: return {"segment_id": seg}

def _lights_and_vp(job):
    d, seg = job
    try:
        G3, G4 = gray_small(d, seg, 3), gray_small(d, seg, 4)
        p0, p1 = match(features(G3), features(G4))
        geo = pair_geometry(p0, p1, K_of(G3)) if len(p0) >= 15 else None
        vp, how = vanishing_point(d, seg, geo["foe"] if geo and geo["par"] >= 2.0 else None)
        return seg, {"lights": light_candidates(d, seg), "vp": [float(vp[0]), float(vp[1])], "vp_how": how}
    except Exception:
        return seg, {"error": traceback.format_exc()[-300:]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lingoqa", required=True); ap.add_argument("--out", default="outputs/cues")
    ap.add_argument("--n_train", type=int, default=15000); ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    data = LingoQA(a.lingoqa)
    official, dev, pool = data.official(), data.dev(), data.training_pool(a.n_train)
    open(f"{a.out}/pool_segments.txt", "w").write("\n".join(pool["segment_id"]))
    print(f"official {len(official)} | dev {len(dev)} | training pool {len(pool)}")

    # 1. motion labels mined from TRAINING answers only (dev clips excluded), at most 600 per label
    tr = data.ac[~data.ac["segment_id"].isin(data.dev_clips())].copy()
    tr["motion"] = [mine_label(q, x) for q, x in zip(tr["question"], tr["answer"])]
    labels = (tr.dropna(subset=["motion"]).groupby("segment_id")["motion"]
                .agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).dropna())
    labels = labels.groupby(labels).head(600)

    # 2. geometry features for labelled + training + test clips
    clips = list(dict.fromkeys([("action", s) for s in labels.index] +
                               list(zip(pool["split"], pool["segment_id"])) +
                               list(zip(official["split"], official["segment_id"])) + list(zip(dev["split"], dev["segment_id"]))))
    with Pool(a.workers) as p:
        feats = list(p.imap_unordered(_features, [(data.dirs[s], c) for s, c in clips], chunksize=16))
    FE = pd.DataFrame(feats).drop_duplicates("segment_id").set_index("segment_id"); FE.to_csv(f"{a.out}/ego_features.csv")

    # 3. motion classifier; labelled clips are described with out-of-fold predictions (no label leakage)
    ego = EgoEstimator().fit(FE, labels); ego.save(f"{a.out}/ego_model.joblib")
    print("motion classifier accuracy (%):", ego.gates)

    # 4. traffic lights + vanishing point for training and test clips
    targets = [c for c in clips if c[1] in set(pool["segment_id"]) | set(official["segment_id"]) | set(dev["segment_id"])]
    with Pool(a.workers) as p:
        rec = dict(p.imap_unordered(_lights_and_vp, [(data.dirs[s], c) for s, c in targets], chunksize=16))
    for seg in rec:
        rec[seg]["state"] = ego.state(FE.loc[seg].to_dict(), segment_id=seg) if seg in FE.index else None
    json.dump({"gates": ego.gates, "records": rec}, open(f"{a.out}/cues_all.json", "w"))
    print(f"cues for {len(rec)} clips | motion stated: {sum(bool(r.get('state')) for r in rec.values())} | "
          f"light candidates: {sum(bool(r.get('lights')) for r in rec.values())} | "
          f"vanishing point: {dict(collections.Counter(r.get('vp_how') for r in rec.values()))}")

if __name__ == "__main__":
    main()

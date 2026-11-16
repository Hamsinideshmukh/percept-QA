"""End to end: clip folder (0.jpg .. 4.jpg) + question -> answer + evidence trace."""
import time
from .router import route
from .cues.geometry import ego_features, gray_small, features, match, pair_geometry, K_of
from .cues.lights import light_candidates
from .cues.road_ahead import vanishing_point
from .cues.ego import EgoEstimator
from .prompt import build, CUE_SET

def extract_cues(clip_dir, ego: EgoEstimator = None):
    """All cues for one clip (CPU, ~1 s). Clip folder = <dir>/<segment_id> with 0.jpg .. 4.jpg."""
    import os
    d, seg = os.path.split(os.path.normpath(clip_dir)); t = time.time()
    feats = ego_features(d, seg) or {}
    state = ego.state(feats) if ego is not None and feats else None
    G3, G4 = gray_small(d, seg, 3), gray_small(d, seg, 4)
    p0, p1 = match(features(G3), features(G4)); geo = pair_geometry(p0, p1, K_of(G3)) if len(p0) >= 15 else None
    vp, how = vanishing_point(d, seg, geo["foe"] if geo and geo["par"] >= 2.0 else None)
    return {"state": state, "lights": light_candidates(d, seg), "vp": [float(vp[0]), float(vp[1])], "vp_how": how,
            "cue_seconds": round(time.time() - t, 2)}

class PerceptQA:
    def __init__(self, adapter, ego_model_path=None, use_cues=True):
        from .model import VLM
        self.vlm = VLM(adapter)
        self.ego = EgoEstimator.load(ego_model_path) if ego_model_path else None
        self.use = CUE_SET if use_cues else ()

    def __call__(self, clip_dir, question, cues=None):
        t = time.time()
        r = route(question)
        cues = cues if cues is not None else extract_cues(clip_dir, self.ego)
        msgs, given, zooms = build(clip_dir, question, r, cues, use=self.use)
        ans = self.vlm(msgs)
        return {"answer": ans, "question_type": r["name"], "cues_used": given, "motion": cues.get("state"),
                "zooms": zooms, "cues": cues, "seconds": round(time.time() - t, 2)}

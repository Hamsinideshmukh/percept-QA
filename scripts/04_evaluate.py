"""Stage 4 - evaluation: answer the official (and optionally dev) questions and score them with Lingo-Judge.

python scripts/04_evaluate.py --lingoqa /path/to/lingoqa --adapter outputs/lora_cues --cues outputs/cues/cues_all.json --out outputs/eval_cues
python scripts/04_evaluate.py --lingoqa /path/to/lingoqa --adapter unsloth/Qwen3.5-2B --out outputs/eval_zero_shot
"""
import argparse, os, json, time
import unsloth
import pandas as pd
from perceptqa.data import LingoQA
from perceptqa.prompt import build, CUE_SET
from perceptqa.model import VLM
from perceptqa.judge import LingoJudge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lingoqa", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cues", default=None, help="cues_all.json; omit to answer from the frames only")
    ap.add_argument("--sets", nargs="+", default=["official"], choices=["official", "dev"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    data = LingoQA(a.lingoqa)
    qs = pd.concat([data.official() if s == "official" else data.dev() for s in a.sets], ignore_index=True)
    rec = json.load(open(a.cues))["records"] if a.cues else {}
    use = CUE_SET if a.cues else ()
    vlm = VLM(a.adapter)

    rows, t = [], time.time()
    for k, r in enumerate(qs.itertuples()):
        msgs, given, _ = build(data.clip_dir(r.split, r.segment_id), r.question, {"cls": r.cls, "cues": r.cues},
                               rec.get(r.segment_id, {}), use=use)
        rows.append({"question_id": r.question_id, "set": r.set, "question_type": r.cname,
                     "cues_used": "|".join(given), "answer": vlm(msgs)})
        if k % 100 == 0: print(f"{k}/{len(qs)} | {(time.time() - t) / (k + 1):.1f} s per question", flush=True)
    preds = pd.DataFrame(rows)

    judge = LingoJudge(); preds["correct"] = 0
    for s in a.sets:
        m = preds["set"] == s
        ok = judge.correct(qs[qs["set"] == s], dict(zip(preds.loc[m, "question_id"], preds.loc[m, "answer"])))
        preds.loc[m, "correct"] = preds.loc[m, "question_id"].map(ok).fillna(0).astype(int)
    preds.to_csv(f"{a.out}/scored.csv", index=False)

    summary = preds.groupby("set")["correct"].mean().mul(100).round(1)
    by_type = preds.pivot_table(index="question_type", columns="set", values="correct", aggfunc="mean").mul(100).round(1)
    report = f"Lingo-Judge accuracy (%)\n{summary.to_string()}\n\nper question type\n{by_type.to_string()}"
    open(f"{a.out}/summary.txt", "w").write(report); print(report)

if __name__ == "__main__":
    main()

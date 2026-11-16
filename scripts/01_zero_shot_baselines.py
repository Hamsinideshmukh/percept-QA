"""Stage 1 - base model selection: zero-shot accuracy of small VLMs on the official LingoQA questions.

python scripts/01_zero_shot_baselines.py --lingoqa /path/to/lingoqa --out outputs/zero_shot
"""
import argparse, os, time, gc
import pandas as pd, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
from perceptqa.data import LingoQA
from perceptqa.judge import LingoJudge

MODELS = {"qwen3.5-2b": "Qwen/Qwen3.5-2B", "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct", "gemma-4-e2b": "google/gemma-4-E2B-it"}

def frames(clip_dir):
    out = []
    for i in range(5):
        im = Image.open(f"{clip_dir}/{i}.jpg").convert("RGB"); im.thumbnail((768, 768)); out.append(im)
    return out

@torch.inference_mode()
def ask(model, processor, clip_dir, question):
    content = [{"type": "image", "image": im} for im in frames(clip_dir)]
    content.append({"type": "text", "text": f"{question}\nAnswer in one short sentence."})
    x = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                      tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=model.dtype)
    o = model.generate(**x, max_new_tokens=64, do_sample=False)
    return processor.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lingoqa", required=True); ap.add_argument("--out", default="outputs/zero_shot")
    ap.add_argument("--models", nargs="+", default=list(MODELS)); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    data = LingoQA(a.lingoqa); qs = data.official().head(a.n)
    answers = {}
    for name in a.models:
        processor = AutoProcessor.from_pretrained(MODELS[name])
        model = AutoModelForMultimodalLM.from_pretrained(MODELS[name], dtype=torch.float16, device_map="cuda").eval()
        t = time.time()
        answers[name] = {r.question_id: ask(model, processor, data.clip_dir(r.split, r.segment_id), r.question) for r in qs.itertuples()}
        print(f"{name}: {(time.time() - t) / len(qs):.1f} s per question")
        pd.Series(answers[name], name="answer").rename_axis("question_id").to_csv(f"{a.out}/preds_{name}.csv")
        del model; gc.collect(); torch.cuda.empty_cache()

    judge = LingoJudge(); rows = []
    for name, ans in answers.items():
        rows.append({"model": name, "lingo_judge": round(100 * judge.correct(qs, ans).mean(), 1), "questions": len(qs)})
    summary = pd.DataFrame(rows); summary.to_csv(f"{a.out}/summary.csv", index=False); print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

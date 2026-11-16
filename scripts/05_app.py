"""Stage 5 - interactive demo: pick an official LingoQA clip, ask a question, see the cues and the answer.

python scripts/05_app.py --lingoqa /path/to/lingoqa --adapter outputs/lora_cues --ego outputs/cues/ego_model.joblib
"""
import argparse, random
import unsloth
import gradio as gr
from PIL import Image
from perceptqa import PerceptQA
from perceptqa.data import LingoQA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lingoqa", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--ego", required=True)
    ap.add_argument("--examples", type=int, default=80); ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    data = LingoQA(a.lingoqa)
    questions = data.official().to_dict("records"); random.Random(0).shuffle(questions)
    examples = {f"{q['question'][:70]}  [{q['segment_id'][:8]}]": q for q in questions[:a.examples]}
    qa = PerceptQA(adapter=a.adapter, ego_model_path=a.ego, use_cues=True)

    def answer(choice, own_question):
        q = examples[choice]; clip = data.clip_dir("evaluation", q["segment_id"])
        question = own_question.strip() or q["question"]
        out = qa(clip, question)
        frames = [(Image.open(f"{clip}/{i}.jpg"), f"t = {i - 4} s") for i in range(5)]
        zooms = [(img, name) for name, img in out["zooms"]]
        trace = (f"question type : {out['question_type']}\n"
                 f"cues used     : {', '.join(out['cues_used']) or 'none'}\n"
                 f"motion        : {out['motion'] or 'not stated'}\n"
                 f"time          : {out['seconds']} s")
        refs = "\n".join(f"- {r}" for r in q["refs"]) if not own_question.strip() else "-"
        return frames, zooms, out["answer"], trace, refs

    with gr.Blocks(title="PerceptQA") as app:
        gr.Markdown("## PerceptQA: driving question answering with classical-CV cues and a fine-tuned 2B VLM")
        with gr.Row():
            choice = gr.Dropdown(list(examples), value=list(examples)[0], label="Clip and question (LingoQA evaluation set)")
            own = gr.Textbox(label="Or ask your own question about this clip")
        run = gr.Button("Answer", variant="primary")
        frames = gr.Gallery(label="Input: 5 frames, 1 second apart", columns=5, height=200)
        with gr.Row():
            zooms = gr.Gallery(label="Zoom crops added by the system", columns=2, height=220)
            with gr.Column():
                ans = gr.Textbox(label="PerceptQA answer")
                trace = gr.Textbox(label="Evidence used", lines=4)
                refs = gr.Textbox(label="Human answers", lines=3)
        run.click(answer, [choice, own], [frames, zooms, ans, trace, refs])
    app.launch(share=a.share)

if __name__ == "__main__":
    main()

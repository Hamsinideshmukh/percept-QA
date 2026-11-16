"""Stage 3 - LoRA fine-tuning of Qwen3.5-2B with Unsloth, with or without cues.

Without --cues : 5 frames + question -> answer
With    --cues : + timestamps, zoom crops, motion sentence; each cue dropped 25% of the time (cue dropout)

python scripts/03_finetune.py --lingoqa /path/to/lingoqa --out outputs/lora_plain
python scripts/03_finetune.py --lingoqa /path/to/lingoqa --cues outputs/cues/cues_all.json --out outputs/lora_cues
"""
import argparse, os, json, random, time, collections
import unsloth                                     # before transformers / trl
import datasets, transformers
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from perceptqa.data import LingoQA
from perceptqa.prompt import build, CUE_SET, TX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lingoqa", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cues", default=None, help="cues_all.json from stage 2; omit for the plain model")
    ap.add_argument("--n_train", type=int, default=15000); ap.add_argument("--p_drop", type=float, default=0.25)
    ap.add_argument("--max_hours", type=float, default=9.0)
    a = ap.parse_args(); t0 = time.time()

    data = LingoQA(a.lingoqa); pool = data.training_pool(a.n_train)
    rec = json.load(open(a.cues))["records"] if a.cues else {}
    rng = random.Random(123)                       # cue dropout decided once per example -> reproducible
    use = [{c for c in CUE_SET if rng.random() >= a.p_drop} if a.cues else set() for _ in range(len(pool))]
    print(f"training examples: {len(pool)} | cues: {'on' if a.cues else 'off'}")

    def make_examples(batch):                      # images are loaded only when the trainer asks for an example
        out = []
        for i in batch["idx"]:
            r = pool.iloc[i]
            msgs, _, _ = build(data.clip_dir(r["split"], r["segment_id"]), r["question"],
                               {"cls": r["cls"], "cues": r["cues"]}, rec.get(r["segment_id"], {}), use=use[i])
            msgs.append({"role": "assistant", "content": [TX(" ".join(str(r["answer"]).split()))]})
            out.append(msgs)
        return {"messages": out}
    train_ds = datasets.Dataset.from_dict({"idx": list(range(len(pool)))}).with_transform(make_examples)

    # 4-bit base model, LoRA on the language layers only; the vision encoder stays frozen
    model, processor = FastVisionModel.from_pretrained("unsloth/Qwen3.5-2B", load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                           finetune_attention_modules=True, finetune_mlp_modules=True,
                                           r=16, lora_alpha=32, lora_dropout=0.0, bias="none", random_state=3407)
    # the loss covers only the answer tokens
    collator = UnslothVisionDataCollator(model, processor, train_on_responses_only=True,
                                         instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

    class TimeLimit(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            print(f"update {state.global_step}/{state.max_steps} | {logs}", flush=True)
            if (time.time() - t0) / 3600 > a.max_hours: control.should_training_stop = True

    FastVisionModel.for_training(model)
    args = SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=1, warmup_steps=10,
                     learning_rate=2e-4, lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.01,
                     fp16=not is_bf16_supported(), bf16=is_bf16_supported(), logging_steps=25, seed=3407,
                     output_dir=f"{a.out}_ckpt", save_strategy="no", report_to="none", remove_unused_columns=False,
                     dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True}, max_length=4096)
    SFTTrainer(model=model, tokenizer=processor, data_collator=collator, train_dataset=train_ds,
               args=args, callbacks=[TimeLimit()]).train()
    model.save_pretrained(a.out); processor.save_pretrained(a.out)
    print(f"adapter saved to {a.out} | {(time.time() - t0) / 3600:.1f} h")

if __name__ == "__main__":
    main()

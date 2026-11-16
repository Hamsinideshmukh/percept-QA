# =====================================================================================
# PerceptQA — NOTEBOOK A: fine-tuning WITHOUT cues ("plain"), Unsloth + LoRA on Qwen3.5-2B
# Kaggle: Accelerator GPU T4 x1 | Internet On | LingoQA dataset attached.  Save Version -> Save & Run All.
# Trains on up to 15,000 questions (one per clip, same order as the cue maker); stops fine-tuning at 8.5 h.
# Output: /kaggle/working/ft_plain_results.tar   (+ /kaggle/working/lora_plain = the adapter)
# =====================================================================================
import os, sys, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less GPU-memory fragmentation
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=False)   # fresh Kaggle machine: install first
import unsloth                     # must come before transformers / trl
import torch, time, json, glob, re, random, gc, tarfile, traceback, math, collections
import pandas as pd, numpy as np
from PIL import Image
print("unsloth", unsloth.__version__, "| torch", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
T0 = time.time()
INPUT, WORK = "/kaggle/input", "/kaggle/working"
MODE = "plain"
OUT = f"{WORK}/ft_{MODE}"; os.makedirs(OUT, exist_ok=True)
def hours(): return (time.time() - T0) / 3600
_log = open(f"{OUT}/log.txt", "a")
def log(*a):
    s = f"[{hours():5.2f}h] " + " ".join(str(x) for x in a); print(s, flush=True); _log.write(s + "\n"); _log.flush()

# ---------------------------------------------------------------- data
ROOT = sorted(glob.glob(f"{INPUT}/**/lingoqa", recursive=True))[0]
def split_dir(s):
    d = f"{ROOT}/{s}/{s}"; return d if os.path.isdir(d) else f"{ROOT}/{s}"
def load(s):
    f = glob.glob(f"{ROOT}/{s}.parquet") + glob.glob(f"{INPUT}/**/{s}.parquet", recursive=True)
    t = pd.read_parquet(f[0]); t["question_id"] = t["question_id"].astype(str); return t
DIRS = {"evaluation": split_dir("evaluation"), "action": split_dir("action"), "scenery": split_dir("scenery")}
ev, ac = load("evaluation"), load("action")
try: sc = load("scenery")
except Exception as e: sc = ac.iloc[:0].copy(); print("no scenery split:", e)
print("rows: evaluation", len(ev), "| action", len(ac), "| scenery", len(sc))


# ---------------------------------------------------------------- question tagger
# =====================================================================================
#                         QUESTION TAGGER v2 (frozen; question text only)
# =====================================================================================
import re

NAMES = {1: "why", 2: "decision", 3: "own_action", 4: "hazard", 5: "layout", 6: "counting", 7: "light_state",
         8: "presence", 9: "where", 10: "others_motion", 11: "identity", 12: "scene_other"}
BASE_CUES = {1: ["EGO_DETAIL", "MOVERS"], 2: ["EGO_DETAIL", "MOVERS"], 3: ["EGO_DETAIL"], 4: ["MOVERS"], 5: [], 6: [],
             7: ["LIGHT_ZOOM"], 8: ["FAR_ZOOM"], 9: [], 10: ["MOVERS"], 11: ["FAR_ZOOM"], 12: []}
NO_FLAG_CUES = {5, 6, 9, 12}

def R(p): return re.compile(p, re.I)
LIGHT = R(r"\btraffic[- ]?(?:lights?|signals?)\b|(?<!brake )(?<!tail )(?<!hazard )(?<!street )(?<!indicator )\blights? (?:is|are|ahead|showing|display\w*|colou?r)\b|\bcolou?r of the lights?\b|\b(?:red|green|amber|yellow) lights?\b")
FAR = R(r"\bsigns?\b|\bspeed limit\b|\bpedestrians?\b|\bcyclists?\b|\bbicycles?\b|\bmotorcycl\w*\b|\bmotorbikes?\b|\bscooters?\b|\bpeople\b")
AGENT = r"(?:pedestrians?|cyclists?|bicycles?|bikes?|motorcycl\w*|motorbikes?|scooters?|cars?|vehicles?|vans?|trucks?|lorry|lorries|bus|buses|taxis?|people|person|riders?|minivans?|sedans?|suvs?|traffic(?! signs?| lights?| signals?| islands?))"
EGO = r"(?:you|we|i|the ego(?: car| vehicle)?|the driver|your car|this car)"

COUNT = R(r"\bhow many\b|\bnumber of\b|\bcount\b")
WHY = R(r"^\s*why\b|\breasons?\b|\bjustif\w*|\bwhat (?:is|was|are) causing\b|\bwhat makes\b|\bwhat caused\b|"
        r"\bhow (?:is|are|does|do) (?:the |a |an )?[\w -]{1,40}? (?:impact\w*|affect\w*|influenc\w*) (?:your|our|the ego)|"
        r"\band why\b|\bexplain\b|\bpurpose of\b|\bin anticipation of\b|\banticipating (?:and|while|by)\b|\badjusting your position\b")
HAZARD = R(r"\bhazards?\b|\bdangers?\b|\bdangerous\b|\brisks?\b|\bpay(?:ing)? attention\b|\battention\b|\bfocus\w*\b|"
           r"\bbe aware\b|\baware of\b|\bwatch(?:ing)? (?:out )?for\b|\bobstacles?\b|\bkeep(?:ing)? an eye\b|\bcautious\b")
DECISION = R(rf"\bshould {EGO}\b|\bis it safe\b|\b(?:can|could) {EGO}\b(?! (?:see|spot|notice|observe|identify|tell|describe))|\bare you (?:permitted|allowed|able)\b|\bmust {EGO}\b|"
             rf"\bdo {EGO} (?:need|have) to\b|\bnecessary\b|\brecommended\b|\bhow should\b|\bbest (?:course of )?action\b|"
             rf"\bcourse of action\b|\bwhat should\b|\bwhat would\b|\bwere to\b|\bwhat if\b|\bis it (?:okay|ok|legal|possible|appropriate)\b|"
             rf"\b(?:road|way|path|lane) clear for\b|\bsafe to\b|\bwould it be\b|\b(?:might |may |will )?(?:have|need) to (?:stop|slow|brake|yield|give way|wait)\b")
OWN = R(rf"\bwhat (?:are|is) {EGO} (?:currently |presently |now )?doing\b|\b(?:current|your|driving|ego(?:'s)?) (?:driving )?actions?\b|"
        rf"\bwhat actions? (?:are|is) {EGO}\b|\b(?:are|is) {EGO} (?:currently |presently )?(?:accelerat|decelerat|slow|stop|start|turn|"
        rf"maintain|brak|mov|driv|overtak|chang|remain|wait|follow|steer|speed)\w*|\bwhat (?:are|is) {EGO} (?:currently )?(?:planning|about) to\b|"
        rf"\bwhat manoeuv\w+|\bwhat maneuver\w*|\bhow (?:are|is) {EGO} (?:driving|moving|positioned)\b|\bwhat (?:is|are) {EGO} up to\b")
OTHERS = R(rf"\bwhich (?:way|direction) (?:is|are) (?:the |a |that |this )?(?:\w+ )?{AGENT}\b|\bwhat (?:is|are) (?:the |a |that |this )?(?:\w+ ){{0,3}}{AGENT} "
           rf"(?:\w+ ){{0,3}}doing\b|\b(?:is|are) (?:the |a |any |that |this )?(?:\w+ ){{0,2}}{AGENT} (?:\w+ ){{0,3}}(?:moving|turning|crossing|stopping|"
           rf"braking|indicating|signalling|signaling|overtaking|reversing|waiting|approaching|pulling|merging|slowing|accelerating|changing|coming|going|about to)\b|"
           rf"\bwhat (?:will|is|are) (?:the |a |that )?(?:\w+ ){{0,2}}{AGENT} (?:\w+ ){{0,2}}(?:do next|going to do|about to do|likely to do)\b|"
           rf"\bwhere (?:is|are) (?:the |a |that )?(?:\w+ ){{0,2}}{AGENT} (?:going|heading|headed|turning)\b|\banticipating from\b|"
           rf"\bwhat (?:is|are) the (?:state|status|behaviou?r|activity) of (?:the |a )?(?:\w+ ){{0,3}}{AGENT}\b")
WHERE = R(r"^\s*where\b|\bwhere (?:is|are|can|do|does)\b|\bwhich (?:lane|side)\b|\b(?:on|from|to) which side\b|\bin which (?:lane|direction|position)\b|"
          r"\bleft or (?:the )?right\b|\bposition of\b|\blocation of\b|\bhow far\b|\bwhich direction (?:is|are) (?:the |a )?(?:[\w ]+ )?(?:located|positioned)\b")
LAYOUT = R(r"\bzebra\b|\b(?:pedestrian|zebra|pelican|level|toucan) crossings?\b(?! (?:the|your|our|this|that|in|on|at|from)\b)|\bcrossings? (?:ahead|nearby|here)\b|"
           r"\b(?:cycle|bike|bus|bicycle|turning|left|right|middle|adjacent|oncoming|same) lanes?\b|\b(?:road |lane )?markings?\b|\b(?:type|kind|width) of (?:the )?lane\b|"
           r"\bbox junction\b|\bjunctions?\b|\bintersections?\b|\broundabouts?\b|\bspeed limit\b|\b(?:road |traffic |street |give[- ]way |stop |warning )?signs?\b|"
           r"\broad ?works?\b|\bbus stops?\b|\bkerbs?\b|\bcurbs?\b|\bpavements?\b|\bone[- ]way\b|\bdouble (?:yellow|white)\b|\b(?:yellow|white) lines?\b|"
           r"\btype of road\b|\bkind of road\b|\broad (?:layout|type|surface)\b|\bislands?\b|\bbollards?\b|\bspeed bumps?\b|\bparking (?:bays?|spaces?)\b")
IDENT = R(r"\bwhat (?:kinds?|types?|sorts?|make|model|colou?rs?|brand)\b|\bwhat'?s the colou?r\b|\bcolou?r of\b|\bwhat (?:vehicles?|objects?)\b|\bwhich (?:vehicles?|objects?|car)\b|"
          r"\bwhat (?:is|are) the (?:vehicles?|objects?|cars?|trucks?|bus)\b|\bidentify\b|\bwhat (?:is|are) (?:directly )?(?:in front|ahead|behind)\b|"
          r"\bwhat (?:is|does) (?:it|that|this) (?:say|read|show)\b|\bwhat (?:is|are) written\b")
PRESENCE = R(rf"^\s*(?:is|are) there\b|^\s*(?:do|can|could) you (?:see|spot|notice|observe)\b|\bany {AGENT}\b|\bis there (?:a|an|any)\b|\bare there any\b|"
             rf"\bis anyone\b|\bis anything\b|\bpresence of\b|\b(?:is|are) (?:the |any )?{AGENT} (?:present|visible|around|nearby|ahead|behind|in front)\b")
LIGHT_STATE = R(r"\bcolou?r\b|\bshow\w*\b|\bindicat\w*\b|\bdisplay\w*\b|\bstate\b|\bstatus\b|\b(?:is|are) (?:the |any )?(?:traffic )?(?:lights?|signals?)\b|"
                r"\bwhat (?:do|does|is|are) the (?:traffic )?(?:lights?|signals?)\b|\bis there (?:a|any) (?:traffic )?(?:lights?|signals?)\b|\bare there (?:any )?(?:traffic )?(?:lights?|signals?)\b|"
                r"\b(?:red|green|amber|yellow)\b")

WHERE_GOING = R(rf"\bwhere (?:is|are) (?:the |a |that )?(?:\w+ ){{0,2}}{AGENT} (?:going|heading|headed|turning)\b")
IDENT_STRONG = R(r"^\s*what (?:kinds?|types?|sorts?|make|model|colou?rs?|brand)\b|^\s*what'?s the colou?r\b|^\s*what (?:is|are) the colou?rs? of\b|^\s*which (?:vehicles?|objects?)\b|^\s*what (?:vehicles?|objects?)\b")

REACT = R(rf"\bhow (?:does|do|did|is|are) {EGO} (?:react\w*|respond\w*|behav\w*)")

def tag(q):
    q = " ".join(str(q).split())
    light, far = bool(LIGHT.search(q)), bool(FAR.search(q))
    if COUNT.search(q): c = 6
    elif re.match(r"\s*how should\b", q, re.I): c = 2
    elif REACT.search(q): c = 3
    elif HAZARD.search(q): c = 4
    elif WHY.search(q): c = 1
    elif DECISION.search(q): c = 2
    elif light and LIGHT_STATE.search(q): c = 7
    elif WHERE_GOING.search(q): c = 10
    elif WHERE.search(q): c = 9
    elif IDENT_STRONG.search(q): c = 11
    elif OTHERS.search(q): c = 10
    elif OWN.search(q): c = 3
    elif LAYOUT.search(q) and not re.search(rf"\b{AGENT} (?:\w+ )?crossing\b", q, re.I): c = 5
    elif IDENT.search(q): c = 11
    elif PRESENCE.search(q): c = 8
    elif LAYOUT.search(q): c = 5
    else: c = 12
    cues = ["TIME"] + BASE_CUES[c]
    if c not in NO_FLAG_CUES:
        if light and "LIGHT_ZOOM" not in cues: cues.append("LIGHT_ZOOM")
        if far and "FAR_ZOOM" not in cues and "LIGHT_ZOOM" not in cues: cues.append("FAR_ZOOM")
    return {"cls": c, "name": NAMES[c], "light": light, "far": far, "cues": cues}




# ---------------------------------------------------------------- test set + training set
N_TRAIN, N_DEV = 15000, 500
has_clip = lambda d, s: all(os.path.exists(f"{d}/{s}/{i}.jpg") for i in range(5))
qs = ev.drop_duplicates("question_id").reset_index(drop=True).assign(split="evaluation", set="eval")
ok_act = sorted(s for s in ac["segment_id"].unique() if has_clip(DIRS["action"], s))
dev_clips = set(random.Random(1).sample(ok_act, min(1000, len(ok_act))))
dev = (ac[ac["segment_id"].isin(dev_clips)].sample(frac=1, random_state=0).drop_duplicates("segment_id")
       .reset_index(drop=True)).head(N_DEV).assign(split="action", set="dev")
items = pd.concat([qs, dev], ignore_index=True)
T = [tag(q) for q in items["question"]]
items["cls"] = [t["cls"] for t in T]; items["cname"] = [t["name"] for t in T]; items["cues"] = [t["cues"] for t in T]

src = pd.concat([ac[~ac["segment_id"].isin(dev_clips)].assign(split="action"), sc.assign(split="scenery")])
src = src[src["answer"].astype(str).str.split().str.len().between(2, 45)]
src = src.sample(frac=1, random_state=7).drop_duplicates("segment_id").reset_index(drop=True)
TS = [tag(q) for q in src["question"]]
src["cls"] = [t["cls"] for t in TS]; src["cname"] = [t["name"] for t in TS]; src["cues"] = [t["cues"] for t in TS]
it = {c: iter(list(g.index)) for c, g in src.groupby("cls")}; order = []
while len(order) < int(N_TRAIN * 1.15) and it:
    for c in list(it):
        try: order.append(next(it[c]))
        except StopIteration: del it[c]
cand = src.loc[order]
cand = cand[[has_clip(DIRS[sp], s) for sp, s in zip(cand["split"], cand["segment_id"])]]
POOL = cand.head(N_TRAIN).reset_index(drop=True)
POOL[["question_id", "segment_id", "split", "question", "answer", "cname"]].to_csv(f"{OUT}/train_pool.csv", index=False)
print("test questions:", len(items), "| training questions:", len(POOL))
print("training mix:", POOL["cname"].value_counts().to_dict())
print("\n3 training examples:"); print(POOL[["question", "answer"]].head(3).to_string())

# ---------------------------------------------------------------- load model
from unsloth import FastVisionModel, is_bf16_supported
MID = "unsloth/Qwen3.5-2B"        # same weights as Qwen/Qwen3.5-2B
model, processor = FastVisionModel.from_pretrained(MID, load_in_4bit=True, use_gradient_checkpointing="unsloth")
print("parameters:", round(sum(p.numel() for p in model.parameters()) / 1e9, 2), "B | GPU memory used:", round(torch.cuda.memory_allocated() / 1e9, 1), "GB")

# ---------------------------------------------------------------- chat format
IM = lambda im: {"type": "image", "image": im}
TX = lambda t: {"type": "text", "text": t}
def frame(d, seg, i):
    im = Image.open(f"{d}/{seg}/{i}.jpg").convert("RGB"); im.thumbnail((512, 512) if i < 4 else (768, 768)); return im
INSTR = "Answer in one short sentence."
def to_messages(row, answer=None):
    d = DIRS[row.split]
    m = [{"role": "user", "content": [IM(frame(d, row.segment_id, i)) for i in range(5)] + [TX(f"{row.question}\n{INSTR}")]}]
    if answer is not None: m.append({"role": "assistant", "content": [TX(" ".join(str(answer).split()))]})
    return m
row = next(POOL.head(1).itertuples())
msgs = to_messages(row, row.answer)
text = processor.apply_chat_template(msgs, tokenize=False)
print(re.sub(r"(<\|image_pad\|>)+", lambda m: f"<|image_pad|> x{len(m.group(0)) // len('<|image_pad|>')}", text))
x = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt")
pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
print("\ntotal tokens:", x["input_ids"].shape[1], "| image tokens:", int((x["input_ids"] == pad).sum()))

# ---------------------------------------------------------------- 3 answers BEFORE fine-tuning
TOK = processor.tokenizer
END_IDS = [i for i in {TOK.convert_tokens_to_ids("<|im_end|>"), TOK.eos_token_id} if isinstance(i, int) and i >= 0]
@torch.inference_mode()
def answer(row):
    x = processor.apply_chat_template(to_messages(row), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    o = model.generate(**x, max_new_tokens=48, do_sample=False, use_cache=True, eos_token_id=END_IDS)
    t = re.sub(r"<think>.*?</think>", "", TOK.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S)
    return re.split(r"<\|im_|\n", t.strip())[0].strip()
FastVisionModel.for_inference(model)
demo_rows = list(items[items["set"] == "dev"].head(3).itertuples())
BEFORE = {r.question_id: answer(r) for r in demo_rows}
for r in demo_rows: print(f"Q: {r.question}\n  human: {r.answer}\n  model (before): {BEFORE[r.question_id]}\n")

# ---------------------------------------------------------------- LoRA + answer-only loss
model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                       finetune_attention_modules=True, finetune_mlp_modules=True,
                                       r=16, lora_alpha=32, lora_dropout=0.0, bias="none", random_state=3407)
model.print_trainable_parameters()

from unsloth.trainer import UnslothVisionDataCollator
try:
    collator = UnslothVisionDataCollator(model, processor, train_on_responses_only=True,
                                         instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    print("collator: loss on the answer only")
except TypeError as e:
    collator = UnslothVisionDataCollator(model, processor); print("WARNING: old unsloth, loss also covers the question:", e)

import datasets
def _build(batch):                                # builds images only when an example is needed -> low RAM
    out = []
    for i in batch["idx"]:
        r = next(POOL.iloc[[i]].itertuples()); out.append(to_messages(r, r.answer))
    return {"messages": out}
TRAIN_DS = datasets.Dataset.from_dict({"idx": list(range(len(POOL)))}).with_transform(_build)
batch = collator([TRAIN_DS[0]])
lab = batch["labels"][0]; kept = lab[lab != -100]
print("tokens in example:", len(lab), "| tokens the model is graded on:", len(kept))
print("graded text:", repr(TOK.decode(kept)))

# ---------------------------------------------------------------- FINE-TUNING
from trl import SFTTrainer, SFTConfig
import transformers
FastVisionModel.for_training(model)
args = SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=1, warmup_steps=10,
                 learning_rate=2e-4, lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.01,
                 fp16=not is_bf16_supported(), bf16=is_bf16_supported(), logging_steps=25, seed=3407,
                 output_dir=f"{WORK}/ckpt_{MODE}", save_strategy="no", report_to="none",
                 remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
                 max_length=4096, dataloader_num_workers=0)
class Progress(transformers.TrainerCallback):
    def on_log(self, a, state, control, logs=None, **kw):
        log(f"update {state.global_step}/{state.max_steps} | {logs}")
        if hours() > 8.5: log("TIME GUARD: stopping early"); control.should_training_stop = True
trainer = SFTTrainer(model=model, tokenizer=processor, data_collator=collator, train_dataset=TRAIN_DS, args=args, callbacks=[Progress()])
t0 = time.time(); trainer.train()
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
TSTAT = {"mode": MODE, "updates": trainer.state.global_step, "minutes": round((time.time() - t0) / 60, 1), "loss_first": losses[:1], "loss_last": losses[-1:]}
log("FINE-TUNED", TSTAT)
model.save_pretrained(f"{WORK}/lora_{MODE}"); processor.save_pretrained(f"{WORK}/lora_{MODE}")   # your adapter: a few tens of MB

# ---------------------------------------------------------------- test: 3 before/after + all 1000
FastVisionModel.for_inference(model)
for r in demo_rows: print(f"Q: {r.question}\n  human: {r.answer}\n  before: {BEFORE[r.question_id]}\n  after:  {answer(r)}\n")
rows = []
for k, r in enumerate(items.itertuples()):
    try: a = answer(r)
    except Exception as e: a = f"__ERROR__ {e!r}"[:150]
    rows.append([r.question_id, r.set, r.segment_id, r.cls, r.cname, "", a])
    if k % 250 == 0: log(f"test {k}/{len(items)} | {a[:70]!r}")
pd.DataFrame(rows, columns=["question_id", "set", "segment_id", "cls", "cname", "cues_given", "answer"]).to_csv(f"{OUT}/preds_ft_plain.csv", index=False)
del model, trainer; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- score with Lingo-Judge
from transformers import pipeline
judge = pipeline("text-classification", model="wayveai/Lingo-Judge", device=0)
REFS = pd.concat([ev[["question_id", "question", "answer"]].assign(set="eval"), dev[["question_id", "question", "answer"]].assign(set="dev")])
p = pd.read_csv(f"{OUT}/preds_ft_plain.csv", dtype={"question_id": str}); p["answer"] = p["answer"].fillna("").astype(str)
rf = REFS.merge(p[["question_id", "set", "answer"]].rename(columns={"answer": "pred"}), on=["question_id", "set"])
texts = [f"[CLS]\nQuestion: {q}\nAnswer: {str(a).lower().strip()}\nStudent: {s.lower().strip()}" for q, a, s in zip(rf["question"], rf["answer"], rf["pred"])]
rf["score"] = [o["score"] for o in judge(texts, batch_size=32, truncation=True, max_length=128)]
best = rf.groupby(["question_id", "set"])["score"].max()
p = p.set_index(["question_id", "set"]); p["correct"] = (best.reindex(p.index).fillna(0) > 0.5).astype(int)
p["words"] = p["answer"].str.split().str.len(); p.reset_index().to_csv(f"{OUT}/scored_ft_plain.csv", index=False)
rep = (f"# Unsloth fine-tuning MODE=plain — {hours():.1f} h\nfine-tuning: {TSTAT}\n"
       f"ft_plain: official {100 * p.xs('eval', level=1)['correct'].mean():.1f} | dev {100 * p.xs('dev', level=1)['correct'].mean():.1f} | words {p['words'].mean():.1f}\n"
       f"(zero-shot: official 64.0 | dev 36.0 | words 14.4)\n\nper question type:\n" + (100 * p.groupby("cname")["correct"].mean()).round(1).to_string())
open(f"{OUT}/REPORT.md", "w").write(rep); print(rep)
with tarfile.open(f"{WORK}/ft_plain_results.tar", "w") as tf:
    for f in glob.glob(f"{OUT}/*"): tf.add(f, arcname=f"ft_plain/{os.path.basename(f)}")
log("ALL DONE -> /kaggle/working/ft_plain_results.tar")
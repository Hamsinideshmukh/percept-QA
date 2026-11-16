# =====================================================================================
# PerceptQA — JUDGE PROBE (~20 min): what does Lingo-Judge reward on why / decision questions?
# Kaggle: GPU T4 (CPU also works, slower) | Internet On | LingoQA dataset attached | no HF token.
# No VLM. Variants built from human reference A, judged against human reference B:
#   action only / reason only / right action + wrong reason / wrong action + right reason / ...
# Output: /kaggle/working/judge_probe_results.tar (report + all scores + eval reference answers)
# =====================================================================================
import os, sys, subprocess, time, json, glob, re, random, gc, tarfile, traceback, math, collections
import torch
T0 = time.time()
INPUT, WORK = "/kaggle/input", "/kaggle/working"
OUT = f"{WORK}/judge_probe"; os.makedirs(OUT, exist_ok=True)
_log = open(f"{OUT}/log.txt", "a")
def hours(): return (time.time() - T0) / 3600
def log(*a):
    s = f"[{hours():5.2f}h] " + " ".join(str(x) for x in a)
    print(s, flush=True); _log.write(s + "\n"); _log.flush()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)

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


# =====================================================================================
#                 CLASSICAL-CV CUES (OpenCV only; no detector, no depth network)

# =====================================================================================
# JUDGE PROBE: what does Lingo-Judge pay for on why / decision questions?
# Variants are built from human reference A and scored ONLY against human reference B.
# =====================================================================================
import pandas as pd
ROOT = sorted(glob.glob(f"{INPUT}/**/lingoqa", recursive=True))[0]
f = glob.glob(f"{ROOT}/evaluation.parquet") + glob.glob(f"{INPUT}/**/evaluation.parquet", recursive=True)
ev = pd.read_parquet(f[0]); ev["question_id"] = ev["question_id"].astype(str)
ev[["question_id", "segment_id", "question", "answer"]].to_csv(f"{OUT}/eval_refs.csv", index=False)   # for offline checks of our cues
g = ev.groupby("question_id")
Q = pd.DataFrame({"question": g["question"].first(), "segment_id": g["segment_id"].first(), "refs": g["answer"].agg(list)})
Q = Q[Q["refs"].str.len() >= 2].copy()
Q["cname"] = [tag(q)["name"] for q in Q["question"]]
Q = Q[Q["cname"].isin(["why", "decision", "own_action"])].copy()
log(f"questions with 2 refs in why/decision/own_action: {len(Q)} {Q['cname'].value_counts().to_dict()}")

SPLIT = re.compile(r"\s*(?:,\s*)?\b(?:because|due to|since|as there|as the|as a|as it|as we|as I|in order to|so that|to avoid|to allow|given that|justified by)\b\s*|\s*,\s+", re.I)
def split_ar(a):
    a = " ".join(str(a).split()).rstrip(".")
    m = SPLIT.search(a)
    if not m: return None
    act, rea = a[:m.start()].strip(" ,"), a[m.end():].strip(" ,")
    return (act, rea, a[m.start():m.end()].strip() or ",") if len(act.split()) >= 1 and len(rea.split()) >= 2 else None
SWAP = [(r"\b(?:stopped|stationary|waiting|remaining stopped|stopping)\b", "accelerating"),
        (r"\b(?:slowing down|decelerating|braking|slowing)\b", "accelerating"),
        (r"\b(?:accelerating|speeding up|driving|proceeding|moving forward|continuing|going straight|maintaining (?:my |a )?(?:constant |steady )?speed)\b", "stopping")]
def wrong_action(act):
    for rx, rep in SWAP:
        if re.search(rx, act, re.I): return re.sub(rx, rep, act, count=1, flags=re.I)
    return None

rows = []
Q = Q.sample(frac=1, random_state=0)
parts = {qid: split_ar(r[0]) for qid, r in zip(Q.index, Q["refs"])}
ok = [qid for qid in Q.index if parts[qid]]
log(f"reference A split into action + reason for {len(ok)} questions")
donor = {qid: ok[(i + 7) % len(ok)] for i, qid in enumerate(ok)}             # a different clip's reference
for qid in ok:
    act, rea, conn = parts[qid]; q = Q.loc[qid, "question"]; refB = Q.loc[qid, "refs"][1]
    dact, drea, _ = parts[donor[qid]]
    V = {"full_A": f"{act} {conn} {rea}" if conn != "," else f"{act}, {rea}",
         "action_only": act, "reason_only": rea,
         "right_action_wrong_reason": f"{act} because {drea}",
         "wrong_action_right_reason": f"{dact} because {rea}",
         "other_clip_answer": Q.loc[donor[qid], "refs"][0]}
    wa = wrong_action(act)
    if wa: V["swapped_motion_verb_right_reason"] = f"{wa} because {rea}"
    for k, v in V.items(): rows.append([qid, Q.loc[qid, "cname"], q, refB, k, v])
P = pd.DataFrame(rows, columns=["question_id", "cname", "question", "refB", "variant", "student"])

from transformers import pipeline
judge = pipeline("text-classification", model="wayveai/Lingo-Judge", device=0 if torch.cuda.is_available() else -1)
texts = [f"[CLS]\nQuestion: {q}\nAnswer: {str(a).lower().strip()}\nStudent: {str(s).lower().strip()}" for q, a, s in zip(P["question"], P["refB"], P["student"])]
P["score"] = [o["score"] for o in judge(texts, batch_size=32, truncation=True, max_length=128)]
P["correct"] = (P["score"] > 0.5).astype(int); P["words"] = P["student"].str.split().str.len()
P.to_csv(f"{OUT}/probe_scored.csv", index=False)

order = ["full_A", "action_only", "reason_only", "right_action_wrong_reason", "wrong_action_right_reason", "swapped_motion_verb_right_reason", "other_clip_answer"]
L = [f"# Judge probe — {hours():.2f} h", f"Built from reference A, judged against reference B only. n questions = {len(ok)}\n",
     "| Variant | n | judged correct % | mean score | words |", "|---|---|---|---|---|"]
for k in order:
    d = P[P["variant"] == k]
    if len(d): L.append(f"| {k} | {len(d)} | {100 * d['correct'].mean():.1f} | {d['score'].mean():.2f} | {d['words'].mean():.1f} |")
L.append("\n## By class (judged correct %)")
L.append(P.pivot_table(index="variant", columns="cname", values="correct", aggfunc="mean").reindex(order).mul(100).round(1).to_string())
L.append("\n## 8 examples")
for qid in ok[:8]:
    d = P[P["question_id"] == qid].set_index("variant")
    L.append(f"\nQ: {Q.loc[qid, 'question']}\n  ref B: {Q.loc[qid, 'refs'][1]}")
    for k in order:
        if k in d.index: L.append(f"  [{d.loc[k, 'correct']}] {k}: {d.loc[k, 'student']}")
rep = "\n".join(L); open(f"{OUT}/REPORT.md", "w").write(rep)
with tarfile.open(f"{WORK}/judge_probe_results.tar", "w") as tf:
    for f in glob.glob(f"{OUT}/*"): tf.add(f, arcname=f"judge_probe/{os.path.basename(f)}")
print("\n" + "=" * 100 + "\n" + rep); log("ALL DONE")

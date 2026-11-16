# =====================================================================================
# PerceptQA — CUE MAKER (Kaggle CPU notebook: Accelerator = None, LingoQA dataset attached)
# Computes the classical-CV cues ONCE for every training clip (same 15,000 as fine-tuning) + the 1000 test questions:
#   ego state sentence (stopped / slowing down / speeding up / turning), traffic-light candidates, road-ahead point.
# Output: /kaggle/working/cues/cues_all.json  -> attached as input to the cues fine-tuning notebook (B).
# =====================================================================================
import os, sys, time, json, glob, re, random, gc, traceback, collections
import pandas as pd, numpy as np
from multiprocessing import Pool
T0 = time.time(); INPUT, WORK = "/kaggle/input", "/kaggle/working"
OUT = f"{WORK}/cues"; os.makedirs(OUT, exist_ok=True)
N_TRAIN, N_DEV, N_EGO_PER_CLASS, N_CPU = 15000, 500, 600, os.cpu_count()
GATE_MIN, CONF_MIN = 70.0, 0.60
def hours(): return (time.time() - T0) / 3600
_log = open(f"{OUT}/log.txt", "a")
def log(*a):
    s = f"[{hours():5.2f}h] " + " ".join(str(x) for x in a); print(s, flush=True); _log.write(s + "\n"); _log.flush()
log("CPUs:", N_CPU)

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
has_clip = lambda d, s: all(os.path.exists(f"{d}/{s}/{i}.jpg") for i in range(5))
qs = ev.drop_duplicates("question_id").reset_index(drop=True).assign(split="evaluation", set="eval")
ok_act = sorted(s for s in ac["segment_id"].unique() if has_clip(DIRS["action"], s))
dev_clips = set(random.Random(1).sample(ok_act, min(1000, len(ok_act))))
dev = (ac[ac["segment_id"].isin(dev_clips)].sample(frac=1, random_state=0).drop_duplicates("segment_id")
       .reset_index(drop=True)).head(N_DEV).assign(split="action", set="dev")
items = pd.concat([qs, dev], ignore_index=True)
src = pd.concat([ac[~ac["segment_id"].isin(dev_clips)].assign(split="action"), sc.assign(split="scenery")])
src = src[src["answer"].astype(str).str.split().str.len().between(2, 45)]
src = src.sample(frac=1, random_state=7).drop_duplicates("segment_id").reset_index(drop=True)
TS = [tag(q) for q in src["question"]]
src["cls"] = [t["cls"] for t in TS]
it = {c: iter(list(g.index)) for c, g in src.groupby("cls")}; order = []
while len(order) < int(N_TRAIN * 1.15) and it:
    for c in list(it):
        try: order.append(next(it[c]))
        except StopIteration: del it[c]
cand = src.loc[order]
cand = cand[[has_clip(DIRS[sp], s) for sp, s in zip(cand["split"], cand["segment_id"])]]
POOL = cand.head(N_TRAIN).reset_index(drop=True)
open(f"{OUT}/pool_segments.txt", "w").write("\n".join(POOL["segment_id"]))
log(f"test questions {len(items)} | training pool {len(POOL)} clips (same order as the fine-tuning notebooks)")

import numpy as np, cv2
cv2.setNumThreads(1)
W_SMALL = 768                     # working width for geometry (frames 0-3 are stored at 768 px)
F_GUESS = 0.55                    # focal length guess = 0.55 * width (no calibration available)
BONNET = 0.80                     # rows below 80 % of the height are the car's own bonnet -> masked out

def gray_small(d, seg, i):
    g = cv2.imread(f"{d}/{seg}/{i}.jpg", cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    if g.shape[1] != W_SMALL: g = cv2.resize(g, (W_SMALL, int(round(g.shape[0] * W_SMALL / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return g

_SIFT = None
def sift():
    global _SIFT
    if _SIFT is None: _SIFT = cv2.SIFT_create(2000)
    return _SIFT

def features(g):
    h, w = g.shape; m = np.full(g.shape, 255, np.uint8); m[int(BONNET * h):, :] = 0
    return sift().detectAndCompute(g, m)

def match(f0, f1):
    (k0, d0), (k1, d1) = f0, f1
    if d0 is None or d1 is None or len(k0) < 20 or len(k1) < 20: return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    good = [a for x in cv2.BFMatcher().knnMatch(d0, d1, k=2) if len(x) == 2 for a in [x[0]] if x[0].distance < 0.75 * x[1].distance]
    return np.float32([k0[m.queryIdx].pt for m in good]), np.float32([k1[m.trainIdx].pt for m in good])

def K_of(g):
    h, w = g.shape; return np.array([[F_GUESS * w, 0, w / 2], [0, F_GUESS * w, h / 2], [0, 0, 1]])

def pair_geometry(p0, p1, K):
    """Essential matrix -> rotation (yaw), translation direction (focus of expansion), derotated parallax, inlier mask."""
    if len(p0) < 15: return None
    E, inl = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3): return None
    _, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=inl.copy())
    inl = inl.ravel() > 0
    p0r = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), K @ R @ np.linalg.inv(K)).reshape(-1, 2)
    par = np.linalg.norm(p1 - p0r, axis=1)
    t = t.ravel(); t = -t if t[2] < 0 else t
    foe = (K @ (t / max(t[2], 1e-6)))[:2] if t[2] > 0.5 * np.linalg.norm(t) else None
    return {"yaw": float(np.rad2deg(np.arctan2(R[0, 2], R[2, 2]))), "par": float(np.median(par[inl])) if inl.sum() > 5 else float(np.median(par)),
            "ninl": int(inl.sum()), "foe": foe, "R": R, "inl": inl, "p0r": p0r}

def road_speed(p0, p1, h, w):
    """Image speed of road texture just above the bonnet (camera height is fixed -> proportional to car speed)."""
    sel = (p0[:, 1] > 0.62 * h) & (p0[:, 1] < BONNET * h) & (np.abs(p0[:, 0] - w / 2) < 0.25 * w)
    return float(np.median(np.linalg.norm(p1[sel] - p0[sel], axis=1))) if sel.sum() >= 5 else np.nan

def ego_features(d, seg):
    """Per-clip geometric features: for each 1-second pair: parallax, yaw, inliers, road speed; 3-frame looming."""
    G = [gray_small(d, seg, i) for i in range(5)]
    if any(g is None for g in G): return None
    h, w = G[0].shape; K = K_of(G[0]); F = [features(g) for g in G]; r = {"segment_id": seg}
    for j in range(4):
        p0, p1 = match(F[j], F[j + 1]); g = pair_geometry(p0, p1, K)
        r[f"par{j}"] = g["par"] if g else np.nan; r[f"yaw{j}"] = g["yaw"] if g else np.nan; r[f"ninl{j}"] = g["ninl"] if g else 0
        r[f"road{j}"] = road_speed(p0, p1, h, w) if len(p0) else np.nan
    for j in range(3):
        s0, s1 = r.get(f"road{j}"), r.get(f"road{j + 1}")
        r[f"roadratio{j}"] = float(np.log((s1 + 0.5) / (s0 + 0.5))) if not (np.isnan(s0) or np.isnan(s1)) else np.nan
    return r

def _ego_worker(args):
    d, seg = args
    try: return ego_features(d, seg)
    except Exception: return {"segment_id": seg}

# ---------------------------------------------------------------- traffic-light candidates (colour + roundness)
def light_candidates(d, seg):
    img = cv2.imread(f"{d}/{seg}/4.jpg")
    if img is None: return []
    H, W = img.shape[:2]; hsv = cv2.cvtColor(img[: int(0.6 * H)], cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]; bright = (s > 120) & (v > 170)
    masks = {"red": bright & ((hch < 10) | (hch > 165)), "amber": bright & (hch >= 12) & (hch < 35), "green": bright & (hch >= 40) & (hch < 95)}
    prev = cv2.imread(f"{d}/{seg}/3.jpg"); sc = prev.shape[1] / W if prev is not None else None
    prev_hsv = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV) if prev is not None else None
    cands = []
    for col, m in masks.items():
        m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 25 or a > 4000: continue
            per = cv2.arcLength(c, True); circ = 4 * np.pi * a / (per * per + 1e-9)
            x, y, bw, bh = cv2.boundingRect(c)
            if circ < 0.55 or not (0.6 < bw / max(bh, 1) < 1.6): continue
            persist = 0.0
            if prev_hsv is not None:   # same colour near the same place one second earlier
                X0, Y0, X1, Y1 = [int(v_ * sc) for v_ in (x - 2 * bw, y - 2 * bh, x + 3 * bw, y + 3 * bh)]
                ph = prev_hsv[max(0, Y0):Y1, max(0, X0):X1]
                if ph.size:
                    pb = (ph[..., 1] > 120) & (ph[..., 2] > 170); hh = ph[..., 0]
                    pm = {"red": (hh < 10) | (hh > 165), "amber": (hh >= 12) & (hh < 35), "green": (hh >= 40) & (hh < 95)}[col] & pb
                    persist = float(pm.any())
            score = circ * (1 + persist) * float(v[y:y + bh, x:x + bw].mean()) / 255
            cands.append({"box": [x, y, x + bw, y + bh], "score": round(score, 3), "persist": persist})
    cands.sort(key=lambda c: -c["score"]); keep = []
    for c in cands:
        if all(abs(c["box"][0] - k["box"][0]) > 150 or abs(c["box"][1] - k["box"][1]) > 150 for k in keep): keep.append(c)
        if len(keep) == 2: break
    return keep

# ---------------------------------------------------------------- vanishing point for the far-ahead crop
def vanishing_point(d, seg, geo_foe=None):
    img = cv2.imread(f"{d}/{seg}/4.jpg", cv2.IMREAD_GRAYSCALE); H, W = img.shape
    if geo_foe is not None:
        x, y = geo_foe[0] * W / W_SMALL, geo_foe[1] * W / W_SMALL
        if 0.2 * W < x < 0.8 * W and 0.25 * H < y < 0.7 * H: return (x, y), "motion"
    small = cv2.resize(img, (W // 2, H // 2)); e = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 60, 160)
    e[: int(0.45 * e.shape[0])] = 0; e[int(BONNET * e.shape[0]):] = 0
    L = cv2.HoughLinesP(e, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=10)
    if L is not None:
        L = np.asarray(L).reshape(-1, 4).astype(np.float64)          # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
        lines = [l for l in L if 0.3 < abs((l[3] - l[1]) / ((l[2] - l[0]) + 1e-6)) < 3][:60]
        pts = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i]; x3, y3, x4, y4 = lines[j]
                den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(den) < 1e-6: continue
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
                if 0.2 * small.shape[1] < px < 0.8 * small.shape[1] and 0.2 * small.shape[0] < py < 0.65 * small.shape[0]: pts.append((px, py))
        if len(pts) >= 3:
            px, py = np.median(np.array(pts), axis=0); return (px * 2, py * 2), "lanes"
    return (0.5 * W, 0.45 * H), "default"

EGO_Q = re.compile(r"what (?:are|is) (?:you|the car|the ego|we)\b.*\bdoing|\bcurrent action\b|\bwhat action\b|\bare you currently\b", re.I)
EGO_L = {"stopped": r"\b(?:stopped|stationary|standing still|at a standstill|waiting|not moving|remain(?:ing)? stopped)\b",
         "slowing": r"\b(?:slow(?:ing)? down|decelerat\w*|brak\w*|reduc\w* (?:my |our )?speed)\b",
         "accelerating": r"\b(?:accelerat\w*|speed(?:ing)? up|increas\w* (?:my |our )?speed|pulling away)\b",
         "turn_left": r"\b(?:turn(?:ing)? left|making a left|taking a left)\b", "turn_right": r"\b(?:turn(?:ing)? right|making a right|taking a right)\b",
         "steady": r"\b(?:maintain\w* (?:a )?(?:constant |steady )?speed|cruis\w*|keep\w* (?:a )?(?:constant|steady) speed)\b"}
EGO_L = {k: re.compile(v, re.I) for k, v in EGO_L.items()}
def ego_label(q, a):
    if not EGO_Q.search(str(q)): return None
    a = str(a)
    if EGO_L["slowing"].search(a) and re.search(r"\bstop(?:ping)?\b", a, re.I): return None
    hits = [k for k, rx in EGO_L.items() if rx.search(a)]
    return hits[0] if len(hits) == 1 else None
tr = ac[~ac["segment_id"].isin(dev_clips)].copy()
tr["ego"] = [ego_label(q, a) for q, a in zip(tr["question"], tr["answer"])]
lab = tr.dropna(subset=["ego"]).groupby("segment_id")["ego"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).dropna()
lab = lab.groupby(lab).head(N_EGO_PER_CLASS)
log(f"ego labels mined from training answers: {lab.value_counts().to_dict()}")

ALL = list(dict.fromkeys(list(zip(items["split"], items["segment_id"])) + list(zip(POOL["split"], POOL["segment_id"]))))
jobs = list(dict.fromkeys([(DIRS["action"], s) for s in lab.index] + [(DIRS[sp], s) for sp, s in ALL]))
log(f"computing ego features for {len(jobs)} clips on {N_CPU} CPUs ...")
feats = []
with Pool(N_CPU) as pool:
    for k, r in enumerate(pool.imap_unordered(_ego_worker, jobs, chunksize=16)):
        if r: feats.append(r)
        if k % 1000 == 0: log(f"  ego features {k}/{len(jobs)}")
FE = pd.DataFrame(feats).drop_duplicates("segment_id").set_index("segment_id")
FCOLS = [c for c in FE.columns if re.match(r"(par|yaw|ninl|road|roadratio)\d", c)]
FE.to_csv(f"{OUT}/ego_features.csv")

from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score as BAC
TRN = FE.loc[FE.index.isin(lab.index)].join(lab.rename("ego"))
TASKS = {"stop":  lambda e: "stopped" if e == "stopped" else ("moving" if e in ("slowing", "accelerating", "steady") else None),
         "trend": lambda e: e if e in ("slowing", "accelerating") else None,
         "turn":  lambda e: "left" if e == "turn_left" else "right" if e == "turn_right" else "no"}
EGO_MODELS, GATE, OOF = {}, {}, {}
for name, f in TASKS.items():
    y = TRN["ego"].map(f); m = y.notna(); X, yy = TRN.loc[m, FCOLS], y[m]
    clf = GBM(max_depth=3, max_iter=250, learning_rate=0.05, class_weight="balanced", random_state=0)
    oof = cross_val_predict(clf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    classes = np.array(sorted(yy.unique()))
    GATE[name] = round(100 * BAC(yy, classes[oof.argmax(1)]), 1); EGO_MODELS[name] = clf.fit(X, yy)
    for s_, pr in zip(X.index, oof): OOF.setdefault(s_, {})[name] = pr      # labelled clips: out-of-fold (no leakage)
log(f"EGO gates (5-fold CV balanced accuracy): {GATE}")
def _pred(name, seg, x):
    clf = EGO_MODELS[name]
    pr = OOF[seg][name] if seg in OOF and name in OOF[seg] else clf.predict_proba(x)[0]
    k = int(np.argmax(pr)); return clf.classes_[k], float(pr[k])
def ego_state(seg):
    if seg not in FE.index: return None
    x = FE.loc[[seg], FCOLS].astype(float); speed = turn = None
    if GATE["stop"] >= GATE_MIN:
        l, c = _pred("stop", seg, x)
        if l == "stopped" and c >= CONF_MIN: speed = "stopped"
        elif l == "moving" and c >= CONF_MIN and GATE["trend"] >= GATE_MIN:
            l2, c2 = _pred("trend", seg, x)
            if c2 >= CONF_MIN: speed = {"slowing": "slowing down", "accelerating": "speeding up"}[l2]
    if GATE["turn"] >= GATE_MIN and speed != "stopped":
        l, c = _pred("turn", seg, x)
        if c >= CONF_MIN: turn = {"left": "turning left", "right": "turning right", "no": "going straight"}[l]
    parts = [p for p in (speed, turn) if p]
    return " and ".join(parts) if parts else None
STATE = {seg: ego_state(seg) for _, seg in ALL}
log("state sentences:", dict(collections.Counter(STATE.values()).most_common(8)))

def rec_worker(args):
    d, seg = args; rec = {}
    try:
        rec["lights"] = light_candidates(d, seg)
        G3, G4 = gray_small(d, seg, 3), gray_small(d, seg, 4)
        p0, p1 = match(features(G3), features(G4)); geo = pair_geometry(p0, p1, K_of(G3)) if len(p0) >= 15 else None
        vp, how = vanishing_point(d, seg, geo["foe"] if geo and geo["par"] >= 2.0 else None)
        rec["vp"] = [float(vp[0]), float(vp[1])]; rec["vp_how"] = how
    except Exception:
        rec["error"] = traceback.format_exc()[-300:]
    return seg, rec
REC = {}
with Pool(N_CPU) as pool:
    for k, (seg, rec) in enumerate(pool.imap_unordered(rec_worker, [(DIRS[sp], s) for sp, s in ALL], chunksize=16)):
        rec["state"] = STATE.get(seg); REC[seg] = rec
        if k % 1000 == 0: log(f"  light / vanishing-point records {k}/{len(ALL)}")
json.dump({"gates": GATE, "gate_min": GATE_MIN, "conf_min": CONF_MIN, "records": REC}, open(f"{OUT}/cues_all.json", "w"), default=str)
log(f"DONE: {len(REC)} clips | state spoken {sum(bool(r.get('state')) for r in REC.values())} | lights {sum(bool(r.get('lights')) for r in REC.values())} | "
    f"vp {dict(collections.Counter(r.get('vp_how') for r in REC.values()))} | errors {sum('error' in r for r in REC.values())}")
log("output: /kaggle/working/cues/cues_all.json")

# =====================================================================================
# PerceptQA — INJECT v2 (no training): EGO state as ONE plain sentence in the SYSTEM prompt + 3 text exemplars
# Kaggle: GPU T4 x1 (the cue maths runs on CPU inside this cell; the GPU is only for Qwen + the judge)
#         Internet On | LingoQA dataset attached | no secrets / no HF token needed
# Run: Save Version -> Save & Run All. ~3.5 h. Report printed at the END of the log + /kaggle/working/inject_v2_results.tar
#
# STEP 1 CPU  EGO state: stop / trend / turn classifiers on keypoint geometry, labels mined from training answers
# STEP 2 CPU  per-clip: state sentence, traffic-light candidates, vanishing point   (MOVERS dropped: 2/8 previews correct)
# STEP 3      layout: system prompt = style + 3 text exemplars + "Right now your car is slowing down and going straight."
#             user turn = frames (+ zoom crops) + question. No <evidence> block, no "uncertain", no codes.
# STEP 4 GPU  runs: base | shots (format control) | ego_noshot | ego | all
# STEP 5      Lingo-Judge, paired tests, copy-rate, per class, 40 previews
# =====================================================================================
import os, sys, subprocess, time, json, glob, re, random, gc, tarfile, traceback, math, collections
T0 = time.time()
INPUT, WORK = "/kaggle/input", "/kaggle/working"
OUT = f"{WORK}/inject_v2"; os.makedirs(OUT, exist_ok=True)
N_DEV = 500                 # held-out dev questions evaluated next to the official 500
N_EGO_PER_CLASS = 600       # training clips per ego label for the EGO classifier
N_CPU = 4
GATE_MIN = 70.0             # an EGO field below this cross-validated balanced accuracy is never spoken
CONF_MIN = 0.60             # per-clip confidence needed before a field is spoken
GEN_DEADLINE_H = 10.5
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

# ---------------------------------------------------------------- MOVERS (independent motion in the last second)
def movers(d, seg):
    G3, G4 = gray_small(d, seg, 3), gray_small(d, seg, 4)
    if G3 is None or G4 is None: return []
    h, w = G3.shape; band = (int(0.30 * h), int(BONNET * h)); K = K_of(G3)
    p0, p1 = match(features(G3), features(G4)); geo = pair_geometry(p0, p1, K) if len(p0) >= 15 else None
    out = []
    def describe(cx, cy, dx, dy):
        side = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "centre"
        near = "near" if cy > 0.58 * h else "far"
        mv = ("moving right→left" if dx < 0 else "moving left→right") if abs(dx) >= abs(dy) else ("coming closer" if dy > 0 else "moving away")
        return f"{mv} at {side}-{near}"
    if geo is None or geo["par"] < 2.0:          # car (almost) stopped: plain frame differencing is reliable
        a = cv2.GaussianBlur(G3, (5, 5), 0); b = cv2.GaussianBlur(G4, (5, 5), 0)
        diff = cv2.absdiff(a, b); diff[:band[0]] = 0; diff[band[1]:] = 0
        m = cv2.morphologyEx((diff > 25).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        for i in sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:3]:
            if st[i, cv2.CC_STAT_AREA] < 250: continue
            fl = flow[lab == i]; dx, dy = float(np.median(fl[:, 0])), float(np.median(fl[:, 1]))
            if abs(dx) + abs(dy) < 0.8: continue
            out.append({"text": describe(cen[i][0], cen[i][1], dx, dy), "box": [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])], "how": "diff"})
    else:                                         # car moving: keypoints that break the car's own motion geometry
        Fm, _ = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.0, 0.999)
        if Fm is None or Fm.shape != (3, 3): return []
        x0 = np.c_[p0, np.ones(len(p0))]; x1 = np.c_[p1, np.ones(len(p1))]
        Fx0 = x0 @ Fm.T; Ftx1 = x1 @ Fm; num = np.sum(x1 * Fx0, axis=1) ** 2
        samp = num / (Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2 + 1e-9)
        res = (p1 - geo["p0r"]); bg = np.median(res[samp < 1.0], axis=0) if (samp < 1.0).sum() > 10 else np.zeros(2)
        out_idx = np.where((samp > 6.0) & (p0[:, 1] > band[0]) & (p0[:, 1] < band[1]))[0]
        cells = {}
        for i in out_idx: cells.setdefault((int(p0[i, 0] // (w / 6)), int(p0[i, 1] // (h / 5))), []).append(i)
        for _, idx in sorted(cells.items(), key=lambda kv: -len(kv[1]))[:3]:
            if len(idx) < 6: continue
            pts = p0[idx]; rel = np.median(res[idx] - bg, axis=0)
            if np.linalg.norm(rel) < 2.0: continue
            x0_, y0_ = pts.min(axis=0); x1_, y1_ = pts.max(axis=0)
            out.append({"text": describe(pts[:, 0].mean(), pts[:, 1].mean(), rel[0], rel[1]),
                        "box": [int(x0_), int(y0_), int(x1_), int(y1_)], "how": "epipolar"})
    return out

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


# =====================================================================================
#        PIPELINE v2: data -> EGO state (CPU) -> inject into system prompt + exemplars -> score
# =====================================================================================
import pandas as pd, torch
from PIL import Image, ImageDraw
from multiprocessing import Pool

ROOT = sorted(glob.glob(f"{INPUT}/**/lingoqa", recursive=True))[0]
def split_dir(s):
    d = f"{ROOT}/{s}/{s}"
    return d if os.path.isdir(d) else f"{ROOT}/{s}"
def load(s):
    f = glob.glob(f"{ROOT}/{s}.parquet") + glob.glob(f"{INPUT}/**/{s}.parquet", recursive=True)
    t = pd.read_parquet(f[0]); t["question_id"] = t["question_id"].astype(str); return t
DIRS = {"evaluation": split_dir("evaluation"), "action": split_dir("action")}
ev, ac = load("evaluation"), load("action")

qs = ev.drop_duplicates("question_id").reset_index(drop=True).assign(split="evaluation", set="eval")
has_clip = lambda d, s: all(os.path.exists(f"{d}/{s}/{i}.jpg") for i in range(5))
ok_act = sorted(s for s in ac["segment_id"].unique() if has_clip(DIRS["action"], s))
dev_clips = set(random.Random(1).sample(ok_act, min(1000, len(ok_act))))
dev = (ac[ac["segment_id"].isin(dev_clips)].sample(frac=1, random_state=0).drop_duplicates("segment_id")
       .reset_index(drop=True)).head(N_DEV).assign(split="action", set="dev")
items = pd.concat([qs, dev], ignore_index=True)
T = [tag(q) for q in items["question"]]
items["cls"] = [t["cls"] for t in T]; items["cname"] = [t["name"] for t in T]; items["cues"] = [t["cues"] for t in T]
log(f"questions: {len(qs)} official + {len(dev)} dev | classes {items['cname'].value_counts().to_dict()}")
items[["question_id", "set", "question", "cls", "cname"]].assign(cues=["|".join(c) for c in items["cues"]]).to_csv(f"{OUT}/tags.csv", index=False)

# =====================================================================================
# STEP 1 (CPU): EGO state. Three BINARY-ish classifiers on keypoint geometry, labels mined from
#               TRAINING answers (never dev/eval clips). A field is spoken only above its gate.
# =====================================================================================
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

eval_clips = list(dict.fromkeys(zip(items["split"], items["segment_id"])))
jobs = [(DIRS["action"], s) for s in lab.index] + [(DIRS[sp], s) for sp, s in eval_clips]
with Pool(N_CPU) as pool:
    feats = [r for r in pool.imap_unordered(_ego_worker, jobs, chunksize=8) if r]
FE = pd.DataFrame(feats).drop_duplicates("segment_id").set_index("segment_id")
log(f"ego features: {len(FE)} clips")
FCOLS = [c for c in FE.columns if re.match(r"(par|yaw|ninl|road|roadratio)\d", c)]

from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score as BAC
TRN = FE.loc[FE.index.isin(lab.index)].join(lab.rename("ego"))
TASKS = {  # name: label map (None = row not used for that task)
    "stop":  lambda e: "stopped" if e == "stopped" else ("moving" if e in ("slowing", "accelerating", "steady") else None),
    "trend": lambda e: e if e in ("slowing", "accelerating") else None,
    "turn":  lambda e: "left" if e == "turn_left" else "right" if e == "turn_right" else "no"}
EGO_MODELS, GATE = {}, {}
for name, f in TASKS.items():
    y = TRN["ego"].map(f); m = y.notna(); X, yy = TRN.loc[m, FCOLS], y[m]
    clf = GBM(max_depth=3, max_iter=250, learning_rate=0.05, class_weight="balanced", random_state=0)
    pred = cross_val_predict(clf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    GATE[name] = round(100 * BAC(yy, pred), 1); EGO_MODELS[name] = clf.fit(X, yy)
log(f"EGO gates (5-fold CV balanced accuracy on training labels): {GATE}  speak only if >= {GATE_MIN} and conf >= {CONF_MIN}")

def _pred(name, x):
    pr = EGO_MODELS[name].predict_proba(x)[0]; k = int(np.argmax(pr))
    return EGO_MODELS[name].classes_[k], float(pr[k])
def ego_state(seg):
    """Plain words for the current state, or None. Nothing is ever written as 'uncertain'."""
    if seg not in FE.index: return None
    x = FE.loc[[seg], FCOLS].astype(float); speed = turn = None
    if GATE["stop"] >= GATE_MIN:
        l, c = _pred("stop", x)
        if l == "stopped" and c >= CONF_MIN: speed = "stopped"
        elif l == "moving" and c >= CONF_MIN and GATE["trend"] >= GATE_MIN:
            l2, c2 = _pred("trend", x)
            if c2 >= CONF_MIN: speed = {"slowing": "slowing down", "accelerating": "speeding up"}[l2]
    if GATE["turn"] >= GATE_MIN and speed != "stopped":
        l, c = _pred("turn", x)
        if c >= CONF_MIN: turn = {"left": "turning left", "right": "turning right", "no": "going straight"}[l]
    parts = [p for p in (speed, turn) if p]
    return " and ".join(parts) if parts else None

# ---- 3 text-only exemplars from TRAINING clips (format alignment; no images, no dev/eval clips)
def _short(a): return 3 <= len(str(a).split()) <= 16
ex_pool = tr[tr["ego"].isin(["stopped", "slowing", "accelerating", "turn_left"]) & tr["answer"].map(_short)].copy()
ex_pool["cls"] = [tag(q)["cls"] for q in ex_pool["question"]]
ex_pool = ex_pool[ex_pool["cls"].isin([1, 2, 3])]
EX = []
for want in ["stopped", "slowing", "accelerating"]:
    c = ex_pool[ex_pool["ego"] == want]
    if len(c): r = c.sample(1, random_state=7).iloc[0]; EX.append((r["question"], r["answer"], {"stopped": "stopped", "slowing": "slowing down", "accelerating": "speeding up"}[want]))
log("exemplars: " + " || ".join(f"[{s}] Q: {q} A: {a}" for q, a, s in EX))

STYLE = "You are the driver. Answer with one short sentence, no preamble."
def system_prompt(state, shots):
    L = [STYLE]
    if shots:
        L.append("Examples of good answers:")
        for q, a, s in EX: L.append((f"(your car is {s}) " if state else "") + f"Q: {q} A: {a}")
    if state: L.append(f"Right now your car is {state}.")
    return "\n".join(L)

# =====================================================================================
# STEP 2 (CPU): per-clip record: ego state, light candidates, vanishing point  (MOVERS dropped: see report)
# =====================================================================================
REC = {}
for k, (sp, seg) in enumerate(eval_clips):
    d = DIRS[sp]; rec = {}
    try:
        rec["state"] = ego_state(seg)
        rec["lights"] = light_candidates(d, seg)
        G3, G4 = gray_small(d, seg, 3), gray_small(d, seg, 4)
        p0, p1 = match(features(G3), features(G4)); geo = pair_geometry(p0, p1, K_of(G3)) if len(p0) >= 15 else None
        vp, how = vanishing_point(d, seg, geo["foe"] if geo and geo["par"] >= 2.0 else None)
        rec["vp"] = [float(vp[0]), float(vp[1])]; rec["vp_how"] = how
    except Exception:
        rec["error"] = traceback.format_exc()[-300:]
    REC[seg] = rec
    if k % 100 == 0: log(f"cue records {k}/{len(eval_clips)}")
json.dump(REC, open(f"{OUT}/cue_records.json", "w"), default=str)
st = {"state_spoken": sum(bool(r.get("state")) for r in REC.values()), "states": dict(collections.Counter(r.get("state") for r in REC.values())),
      "light_candidates": sum(bool(r.get("lights")) for r in REC.values()), "vp_sources": dict(collections.Counter(r.get("vp_how") for r in REC.values())),
      "errors": sum("error" in r for r in REC.values())}
log(f"cue records done for {len(REC)} clips: {st}")

# =====================================================================================
# STEP 3: message layout.  Cue TEXT lives in the SYSTEM prompt (one plain sentence), never in a
#         labelled evidence block next to the question. Zoom crops stay in the user turn.
# =====================================================================================
IM = lambda im: {"type": "image", "image": im}
TX = lambda t: {"type": "text", "text": t}
def frame(d, seg, i):
    im = Image.open(f"{d}/{seg}/{i}.jpg").convert("RGB"); im.thumbnail((512, 512) if i < 4 else (768, 768)); return im
def crop_full(d, seg, cx, cy, wc, hc, out):
    img = Image.open(f"{d}/{seg}/4.jpg").convert("RGB"); W, H = img.size; wc, hc = min(wc, W), min(hc, H)
    x0 = int(min(max(cx - wc / 2, 0), W - wc)); y0 = int(min(max(cy - hc / 2, 0), H - hc))
    return img.crop((x0, y0, x0 + wc, y0 + hc)).resize(out, Image.LANCZOS)
def light_crops(d, seg):
    out = []
    for c in REC.get(seg, {}).get("lights", [])[:2]:
        x0, y0, x1, y1 = c["box"]; side = max(240, 7 * max(x1 - x0, y1 - y0))
        out.append(crop_full(d, seg, (x0 + x1) / 2, (y0 + y1) / 2, side, side, (336, 336)))
    return out
def far_crop(d, seg):
    vp = REC.get(seg, {}).get("vp")
    return crop_full(d, seg, vp[0], vp[1], 960, 540, (448, 252)) if vp else None

PREAMBLE = "5 frames from your car's front camera, 1 second apart, oldest first. The last frame is now."
# run config: shots in {None,"plain","ctx"}, state (bool, routed to EGO classes), zoom (bool, routed), time (bool, all)
RUNS = {"base":       dict(shots=None,    state=False, zoom=False, time=False),
        "shots":      dict(shots="plain", state=False, zoom=False, time=False),   # control: format only, no cue info
        "ego_noshot": dict(shots=None,    state=True,  zoom=False, time=False),   # state sentence alone
        "ego":        dict(shots="ctx",   state=True,  zoom=False, time=False),   # state sentence + exemplars
        "all":        dict(shots="ctx",   state=True,  zoom=True,  time=True)}
def build(row, cfg):
    d, seg = DIRS[row.split], row.segment_id; r = REC.get(seg, {}); routed = set(row.cues); given = []
    c = []
    if cfg["time"]:
        c.append(TX(PREAMBLE)); given.append("TIME")
        for i in range(5): c += [TX(f"t = {i - 4} s" + (" (now)" if i == 4 else "") + ":"), IM(frame(d, seg, i))]
    else:
        c += [IM(frame(d, seg, i)) for i in range(5)]
    if cfg["zoom"] and "LIGHT_ZOOM" in routed and r.get("lights"):
        for im in light_crops(d, seg): c += [TX("Zoomed view of a possible traffic light in the last frame:"), IM(im)]
        given.append("LIGHT_ZOOM")
    if cfg["zoom"] and "FAR_ZOOM" in routed and "LIGHT_ZOOM" not in given:
        im = far_crop(d, seg)
        if im is not None: c += [TX("Zoomed view of the road far ahead in the last frame:"), IM(im)]; given.append("FAR_ZOOM")
    state = r.get("state") if (cfg["state"] and "EGO_DETAIL" in routed) else None
    if state: given.append("EGO")
    if cfg["shots"]: given.append("SHOTS")
    sysm = system_prompt(state, cfg["shots"]) if (state or cfg["shots"]) else None
    return sysm, c, given

# =====================================================================================
# STEP 4 (GPU): untrained Qwen3.5-2B
# =====================================================================================
from transformers import AutoProcessor, AutoModelForMultimodalLM
MID = "Qwen/Qwen3.5-2B"
PROC = AutoProcessor.from_pretrained(MID)
MODEL = AutoModelForMultimodalLM.from_pretrained(MID, dtype=torch.float16, device_map="cuda").eval()
TOK = PROC.tokenizer
STOP = [i for i in {TOK.convert_tokens_to_ids("<|im_end|>"), TOK.eos_token_id} if isinstance(i, int) and i >= 0]
@torch.inference_mode()
def ask(sysm, content, question):
    msgs = ([{"role": "system", "content": [TX(sysm)]}] if sysm else []) + [{"role": "user", "content": content + [TX(f"{question}\nAnswer in one short sentence.")]}]
    x = PROC.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(MODEL.device, dtype=MODEL.dtype)
    o = MODEL.generate(**x, max_new_tokens=64, do_sample=False, eos_token_id=STOP)
    return PROC.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()

BASE = {}
for name, cfg in RUNS.items():
    path = f"{OUT}/preds_{name}.csv"
    if hours() > GEN_DEADLINE_H: log("TIME GUARD: skip", name); continue
    rows, gen = [], 0
    for k, r in enumerate(items.itertuples()):
        key = (r.question_id, r.set)
        try:
            sysm, content, given = build(r, cfg)
            if name != "base" and not given: a = BASE[key]          # nothing routed/available -> identical input to base
            else: a = ask(sysm, content, r.question); gen += 1
        except Exception as e:
            a, given = f"__ERROR__ {e!r}"[:150], ["error"]
        if name == "base": BASE[key] = a
        rows.append([r.question_id, r.set, r.segment_id, r.cls, r.cname, "|".join(given), a])
        if k % 250 == 0: log(f"{name} {k}/{len(items)} generated={gen} | {a[:60]!r}")
    pd.DataFrame(rows, columns=["question_id", "set", "segment_id", "cls", "cname", "cues_given", "answer"]).to_csv(path, index=False)
    log(f"SAVED {name}: {gen} new answers")
del MODEL; gc.collect(); torch.cuda.empty_cache()

# =====================================================================================
# STEP 5: Lingo-Judge, paired tests, copy-rate, per class
# =====================================================================================
from transformers import pipeline
judge = pipeline("text-classification", model="wayveai/Lingo-Judge", device=0)
REFS = pd.concat([ev[["question_id", "question", "answer"]].assign(set="eval"), dev[["question_id", "question", "answer"]].assign(set="dev")])
from math import comb
def mcn(a, b):
    bw, aw = int(((b == 1) & (a == 0)).sum()), int(((a == 1) & (b == 0)).sum()); n = bw + aw
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(bw, aw) + 1)) / 2.0 ** n)
    return round(100 * (b.mean() - a.mean()), 1), bw, aw, round(p, 4)
COPY = re.compile(r"right now|your car is|slowing down|speeding up|going straight|based on|visual evidence|ego.vehicle", re.I)
S = {}
for name in RUNS:
    path = f"{OUT}/preds_{name}.csv"
    if not os.path.exists(path): continue
    p = pd.read_csv(path, dtype={"question_id": str}); p["answer"] = p["answer"].fillna("").astype(str)
    rf = REFS.merge(p[["question_id", "set", "answer"]].rename(columns={"answer": "pred"}), on=["question_id", "set"])
    texts = [f"[CLS]\nQuestion: {q}\nAnswer: {str(a).lower().strip()}\nStudent: {s_.lower().strip()}" for q, a, s_ in zip(rf["question"], rf["answer"], rf["pred"])]
    rf["score"] = [o["score"] for o in judge(texts, batch_size=32, truncation=True, max_length=128)]
    best = rf.groupby(["question_id", "set"])["score"].max()
    p = p.set_index(["question_id", "set"]); p["correct"] = (best.reindex(p.index).fillna(0) > 0.5).astype(int)
    p["words"] = p["answer"].str.split().str.len(); p["copy"] = p["answer"].str.contains(COPY).astype(int)
    p.reset_index().to_csv(path.replace("preds_", "scored_"), index=False); S[name] = p

L = [f"# Inject v2 (untrained Qwen3.5-2B): cue sentence in the system prompt + text exemplars — {hours():.1f} h\n",
     f"EGO gates (5-fold CV balanced acc on training labels): {GATE}; spoken only if gate >= {GATE_MIN} and conf >= {CONF_MIN}",
     f"Records: {st}", "Exemplars: " + " || ".join(f"[{s}] Q: {q} A: {a}" for q, a, s in EX),
     "\nMOVERS dropped before this run: by eye, boxes were on real movers in 2 of 8 preview clips (lane paint, ledges, windows otherwise).\n",
     "Runs: shots = exemplars only (format control, no cue info) | ego_noshot = state sentence only | ego = state + exemplars | all = + zooms + timestamps.",
     "Each run vs base, only on the questions that received something. Paired exact McNemar.\n",
     "| Run | Questions changed | Base acc | New acc | Change | wins/losses | p | words base→new | copy-rate base→new |", "|---|---|---|---|---|---|---|---|---|"]
rows = []
if "base" in S:
    B = S["base"]
    for name in RUNS:
        if name == "base" or name not in S: continue
        A, C = B.align(S[name], join="inner", axis=0); g = (C["cues_given"].fillna("") != "").values
        if g.sum() < 5: L.append(f"| {name} | {int(g.sum())} | | | | | | | |"); continue
        d_, bw, aw, pv = mcn(A["correct"].values[g], C["correct"].values[g])
        L.append(f"| {name} | {int(g.sum())} | {100 * A['correct'].values[g].mean():.1f} | {100 * C['correct'].values[g].mean():.1f} | {d_:+.1f} | {bw}/{aw} | {pv} | "
                 f"{A['words'].values[g].mean():.1f}→{C['words'].values[g].mean():.1f} | {100 * A['copy'].values[g].mean():.0f}%→{100 * C['copy'].values[g].mean():.0f}% |")
        for cn in sorted(A["cname"].unique()):
            m = g & (A["cname"] == cn).values
            if m.sum() >= 10:
                d2, b2, a2, p2 = mcn(A["correct"].values[m], C["correct"].values[m])
                rows.append({"run": name, "class": cn, "n": int(m.sum()), "delta": d2, "wins": b2, "losses": a2, "p": p2})
        for s_ in ["eval", "dev"]:
            m = g & (A.index.get_level_values(1) == s_).values
            if m.sum() >= 10: rows.append({"run": name, "class": f"SET:{s_}", "n": int(m.sum()), "delta": mcn(A["correct"].values[m], C["correct"].values[m])[0], "wins": "", "losses": "", "p": mcn(A["correct"].values[m], C["correct"].values[m])[3]})
    # the key contrast: ego vs shots on the same EGO-routed questions (does the STATE add anything beyond format?)
    if "ego" in S and "shots" in S:
        A, C = S["shots"].align(S["ego"], join="inner", axis=0); g = C["cues_given"].str.contains("EGO").fillna(False).values
        d_, bw, aw, pv = mcn(A["correct"].values[g], C["correct"].values[g])
        L.append(f"\n**ego vs shots** on the {int(g.sum())} EGO-routed questions (state information beyond format): {d_:+.1f}, wins/losses {bw}/{aw}, p={pv}")
    L.append(f"\nBase accuracy: official 500 = {100 * B.xs('eval', level=1)['correct'].mean():.1f}, dev = {100 * B.xs('dev', level=1)['correct'].mean():.1f}")
pt = pd.DataFrame(rows); pt.to_csv(f"{OUT}/by_class.csv", index=False)
if len(pt):
    L.append("\n## Per question class / set (n >= 10)")
    for r in pt.sort_values(["run", "delta"], ascending=[True, False]).itertuples():
        L.append(f"- {r.run} on {r[2]}: {r.delta:+.1f} (n={r.n}, {r.wins}/{r.losses}, p={r.p})")

# ---- 40 previews: last frame + state sentence + light boxes + VP (check the EGO sentence by eye)
os.makedirs(f"{OUT}/previews", exist_ok=True)
for sp, seg in eval_clips[:40]:
    try:
        d = DIRS[sp]; r = REC.get(seg, {})
        ims = [Image.open(f"{d}/{seg}/{i}.jpg").convert("RGB") for i in (2, 4)]
        for im in ims: im.thumbnail((640, 640))
        sheet = Image.new("RGB", (ims[0].width + ims[1].width + 10, max(ims[0].height, ims[1].height) + 24), (0, 0, 0))
        sheet.paste(ims[0], (0, 24)); sheet.paste(ims[1], (ims[0].width + 10, 24))
        dr = ImageDraw.Draw(sheet); dr.text((4, 4), f"t=-2s | t=0s   state: {r.get('state')}   vp: {r.get('vp_how')}", fill=(255, 255, 0))
        sheet.save(f"{OUT}/previews/{seg}.jpg", quality=80)
    except Exception: pass
rep = "\n".join(L); open(f"{OUT}/REPORT.md", "w").write(rep)
with tarfile.open(f"{WORK}/inject_v2_results.tar", "w") as tf:
    for f in glob.glob(f"{OUT}/*.csv") + glob.glob(f"{OUT}/*.md") + glob.glob(f"{OUT}/*.txt") + glob.glob(f"{OUT}/*.json") + glob.glob(f"{OUT}/previews/*"):
        tf.add(f, arcname=f"inject_v2/{os.path.relpath(f, OUT)}")
print("\n" + "=" * 100 + "\nCOPY EVERYTHING BELOW IF FILE UPLOAD FAILS\n" + "=" * 100 + "\n" + rep)
log("ALL DONE")

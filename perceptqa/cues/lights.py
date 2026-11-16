"""Traffic-light candidates: bright saturated red / amber / green blobs, round, persistent across frames."""
import numpy as np, cv2

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


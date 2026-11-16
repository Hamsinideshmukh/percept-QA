"""Keypoint geometry between consecutive frames (OpenCV only): rotation, parallax, road-texture speed."""
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


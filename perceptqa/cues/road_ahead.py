"""Vanishing point = where the road disappears ahead: focus of expansion when moving, lane-line intersections otherwise."""
import numpy as np, cv2
from .geometry import W_SMALL, BONNET

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

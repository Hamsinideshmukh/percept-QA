"""Ego-motion state from geometry features: three small gradient-boosted classifiers (stopped, slowing vs speeding up, turning).
Trained on labels mined from LingoQA TRAINING answers; each field is spoken only if its cross-validated accuracy >= gate."""
import re, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as GBM
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score as BAC

EGO_Q = re.compile(r"what (?:are|is) (?:you|the car|the ego|we)\b.*\bdoing|\bcurrent action\b|\bwhat action\b|\bare you currently\b", re.I)
EGO_L = {k: re.compile(v, re.I) for k, v in {
    "stopped": r"\b(?:stopped|stationary|standing still|at a standstill|waiting|not moving|remain(?:ing)? stopped)\b",
    "slowing": r"\b(?:slow(?:ing)? down|decelerat\w*|brak\w*|reduc\w* (?:my |our )?speed)\b",
    "accelerating": r"\b(?:accelerat\w*|speed(?:ing)? up|increas\w* (?:my |our )?speed|pulling away)\b",
    "turn_left": r"\b(?:turn(?:ing)? left|making a left|taking a left)\b",
    "turn_right": r"\b(?:turn(?:ing)? right|making a right|taking a right)\b",
    "steady": r"\b(?:maintain\w* (?:a )?(?:constant |steady )?speed|cruis\w*|keep\w* (?:a )?(?:constant|steady) speed)\b"}.items()}
TASKS = {"stop": lambda e: "stopped" if e == "stopped" else ("moving" if e in ("slowing", "accelerating", "steady") else None),
         "trend": lambda e: e if e in ("slowing", "accelerating") else None,
         "turn": lambda e: "left" if e == "turn_left" else "right" if e == "turn_right" else "no"}
FCOL = re.compile(r"(par|yaw|ninl|road|roadratio)\d")

def mine_label(question, answer):
    """One motion label from a training answer, or None if the answer is ambiguous."""
    if not EGO_Q.search(str(question)): return None
    a = str(answer)
    if EGO_L["slowing"].search(a) and re.search(r"\bstop(?:ping)?\b", a, re.I): return None
    hits = [k for k, rx in EGO_L.items() if rx.search(a)]
    return hits[0] if len(hits) == 1 else None

class EgoEstimator:
    def __init__(self, gate=70.0, conf=0.60):
        self.gate, self.conf, self.models, self.gates, self.cols = gate, conf, {}, {}, None
        self.oof = {}      # segment_id -> {task: class probabilities} for clips whose label was used in training

    def fit(self, features: pd.DataFrame, labels: pd.Series):
        """features: one row per clip (index = segment_id); labels: mined motion label per clip."""
        self.cols = [c for c in features.columns if FCOL.match(c)]
        T = features.loc[features.index.isin(labels.index), self.cols].join(labels.rename("ego"))
        for name, f in TASKS.items():
            y = T["ego"].map(f); m = y.notna(); X, yy = T.loc[m, self.cols], y[m]
            clf = GBM(max_depth=3, max_iter=250, learning_rate=0.05, class_weight="balanced", random_state=0)
            oof = cross_val_predict(clf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
            self.gates[name] = round(100 * BAC(yy, np.array(sorted(yy.unique()))[oof.argmax(1)]), 1)
            self.models[name] = clf.fit(X, yy)
            for seg, pr in zip(X.index, oof): self.oof.setdefault(seg, {})[name] = pr
        return self

    def _pred(self, name, x, seg=None):
        pr = self.oof[seg][name] if seg in self.oof and name in self.oof[seg] else self.models[name].predict_proba(x)[0]
        k = int(np.argmax(pr)); return self.models[name].classes_[k], float(pr[k])

    def state(self, feat_row: dict, segment_id=None):
        """Plain words ('slowing down', 'stopped', ...) or None. A field is never spoken below its gate or confidence."""
        x = pd.DataFrame([{c: feat_row.get(c, np.nan) for c in self.cols}]).astype(float); speed = turn = None
        if self.gates.get("stop", 0) >= self.gate:
            l, c = self._pred("stop", x, segment_id)
            if l == "stopped" and c >= self.conf: speed = "stopped"
            elif l == "moving" and c >= self.conf and self.gates.get("trend", 0) >= self.gate:
                l2, c2 = self._pred("trend", x, segment_id)
                if c2 >= self.conf: speed = {"slowing": "slowing down", "accelerating": "speeding up"}[l2]
        if self.gates.get("turn", 0) >= self.gate and speed != "stopped":
            l, c = self._pred("turn", x, segment_id)
            if c >= self.conf: turn = {"left": "turning left", "right": "turning right", "no": "going straight"}[l]
        parts = [p for p in (speed, turn) if p]
        return " and ".join(parts) if parts else None

    def save(self, path):
        import joblib; joblib.dump(self, path)
    @staticmethod
    def load(path):
        import joblib; return joblib.load(path)

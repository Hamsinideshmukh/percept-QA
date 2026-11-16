"""LingoQA loading and the fixed splits shared by every stage.

official : the 500 evaluation questions (2 human answers each)
dev      : 500 held-out questions from 1,000 held-out training clips
pool     : up to 15,000 training questions, one per clip, balanced over question types
All splits are deterministic (fixed seeds), so every script sees exactly the same clips.
"""
import os, random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .router import route

def split_dir(root, split):
    """Frame folder of a split: <root>/<split>/<split>/<clip>/0..4.jpg (or one level less)."""
    d = f"{root}/{split}/{split}"
    return d if os.path.isdir(d) else f"{root}/{split}"

def load_table(root, split):
    """Question/answer table of a split (evaluation, action, scenery)."""
    for p in [f"{root}/{split}.parquet", f"{root}/{split}/{split}.parquet", f"{root}/{split}/{split}/{split}.parquet"]:
        if os.path.exists(p):
            t = pd.read_parquet(p); t["question_id"] = t["question_id"].astype(str); return t
    raise FileNotFoundError(f"{split}.parquet not found under {root}")

def has_clips(pairs, workers=32):
    """For each (frame_dir, clip_id): do all 5 frames exist? Threaded, because the disk is the bottleneck."""
    ok = lambda p: all(os.path.exists(f"{p[0]}/{p[1]}/{i}.jpg") for i in range(5))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(ok, pairs))

def add_routing(df):
    """Question type and allowed cues for every row (see router.py)."""
    r = [route(q) for q in df["question"]]
    return df.assign(cls=[x["cls"] for x in r], cname=[x["name"] for x in r], cues=[x["cues"] for x in r])

class LingoQA:
    def __init__(self, root):
        self.root = root
        self.dirs = {s: split_dir(root, s) for s in ("evaluation", "action", "scenery")}
        self.ev, self.ac = load_table(root, "evaluation"), load_table(root, "action")
        try:
            self.sc = load_table(root, "scenery")
        except FileNotFoundError:
            self.sc = self.ac.iloc[:0].copy()
        self._dev_clips = None

    def clip_dir(self, split, segment_id):
        return f"{self.dirs[split]}/{segment_id}"

    def official(self):
        """500 official questions, one row each, with the list of human answers."""
        g = self.ev.groupby("question_id").agg(question=("question", "first"), segment_id=("segment_id", "first"),
                                                refs=("answer", list)).reset_index()
        return add_routing(g.assign(split="evaluation", set="official"))

    def dev_clips(self):
        """1,000 held-out training clips (seed 1). Never used for training."""
        if self._dev_clips is None:
            segs = sorted(self.ac["segment_id"].unique())
            ok = has_clips([(self.dirs["action"], s) for s in segs])
            ok_segs = [s for s, k in zip(segs, ok) if k]
            self._dev_clips = set(random.Random(1).sample(ok_segs, min(1000, len(ok_segs))))
        return self._dev_clips

    def dev(self, n=500):
        """500 dev questions, one per held-out clip (seed 0)."""
        d = (self.ac[self.ac["segment_id"].isin(self.dev_clips())].sample(frac=1, random_state=0)
             .drop_duplicates("segment_id").reset_index(drop=True).head(n))
        return add_routing(d.assign(split="action", set="dev", refs=[[a] for a in d["answer"]]))

    def training_pool(self, n=15000):
        """One question per clip, answers of 2-45 words, round-robin over the 12 question types (seed 7)."""
        src = pd.concat([self.ac[~self.ac["segment_id"].isin(self.dev_clips())].assign(split="action"),
                         self.sc.assign(split="scenery")])
        src = src[src["answer"].astype(str).str.split().str.len().between(2, 45)]
        src = add_routing(src.sample(frac=1, random_state=7).drop_duplicates("segment_id").reset_index(drop=True))
        queues = {c: iter(list(g.index)) for c, g in src.groupby("cls")}; order = []
        while len(order) < int(n * 1.15) and queues:
            for c in list(queues):
                try: order.append(next(queues[c]))
                except StopIteration: del queues[c]
        cand = src.loc[order]
        cand = cand[has_clips(list(zip([self.dirs[s] for s in cand["split"]], cand["segment_id"])))]
        return cand.head(n).reset_index(drop=True)

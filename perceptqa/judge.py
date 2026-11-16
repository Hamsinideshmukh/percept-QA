"""Lingo-Judge scoring: an answer is correct if the judge's score against ANY human answer is > 0.5."""
import pandas as pd

class LingoJudge:
    def __init__(self, device=0):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model="wayveai/Lingo-Judge", device=device)

    def correct(self, questions: pd.DataFrame, answers: dict):
        """questions: rows with question_id, question, refs (list of human answers).
        answers: {question_id: model answer}. Returns a Series question_id -> 0/1."""
        rows = [(r.question_id, r.question, ref, str(answers.get(r.question_id, "")))
                for r in questions.itertuples() for ref in r.refs]
        texts = [f"[CLS]\nQuestion: {q}\nAnswer: {str(ref).lower().strip()}\nStudent: {a.lower().strip()}"
                 for _, q, ref, a in rows]
        scores = [o["score"] for o in self.pipe(texts, batch_size=32, truncation=True, max_length=128)]
        best = pd.DataFrame({"question_id": [r[0] for r in rows], "score": scores}).groupby("question_id")["score"].max()
        return (best > 0.5).astype(int)

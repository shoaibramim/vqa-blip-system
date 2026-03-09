"""
evaluator.py

Computes evaluation metrics for the VQA classification task:

    - Accuracy  : Exact match between predicted and true class labels.
    - F1 Score  : Weighted F1 using sklearn (handles class imbalance).
    - BLEU Score: Sentence-level BLEU treating each answer string as a
                  word sequence (useful when answers are multi-word phrases).
"""

from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Evaluator:
    """
    Stateless evaluator that computes classification and token-level
    generation metrics from lists of predicted and reference class indices.
    """

    def compute(
        self,
        predictions: List[int],
        references: List[int],
        questions: Optional[List[str]] = None,
        label2answer: Optional[Dict[int, str]] = None,
    ) -> Dict[str, float]:
        """
        Compute accuracy, weighted F1, and BLEU for a set of predictions.

        Parameters
        ----------
        predictions  : Predicted class indices (list of ints).
        references   : Ground-truth class indices (list of ints).
        questions    : Optional question strings (reserved for future metrics).
        label2answer : Optional mapping from label index to answer string.
                       When provided, BLEU is computed over answer words;
                       otherwise it falls back to exact-match BLEU.

        Returns
        -------
        dict with keys "accuracy", "f1", "bleu" (all floats, 4 d.p.).
        """
        if len(predictions) == 0 or len(references) == 0:
            return {"accuracy": 0.0, "f1": 0.0, "bleu": 0.0}

        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(
            references, predictions, average="weighted", zero_division=0
        )
        bleu = self._compute_bleu(predictions, references, label2answer)

        return {
            "accuracy": round(float(accuracy), 4),
            "f1": round(float(f1), 4),
            "bleu": round(float(bleu), 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_bleu(
        self,
        predictions: List[int],
        references: List[int],
        label2answer: Optional[Dict[int, str]],
    ) -> float:
        """
        Compute sentence BLEU treating each answer as a tokenised string.

        Underscores in answer strings (e.g., "garbage_bin") are replaced
        with spaces before tokenisation so multi-word answers are handled
        as proper word sequences.
        """
        smoother = SmoothingFunction().method1
        scores: List[float] = []

        for pred, ref in zip(predictions, references):
            if label2answer is not None:
                pred_tokens = (
                    label2answer.get(pred, str(pred)).replace("_", " ").split()
                )
                ref_tokens = (
                    label2answer.get(ref, str(ref)).replace("_", " ").split()
                )
            else:
                pred_tokens = [str(pred)]
                ref_tokens = [str(ref)]

            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=smoother,
            )
            scores.append(score)

        return sum(scores) / max(len(scores), 1)

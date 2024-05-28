import numpy as np
from copy import deepcopy
from typing import Dict, List, Any
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score


def _transform(
    y_pred: List[List],
    y_true: List[List],
):
    mlb = MultiLabelBinarizer()
    tmp = deepcopy(y_true)
    tmp.extend(y_pred)
    mlb.fit(tmp)
    y_true_binary = mlb.transform(y_true)
    y_pred_binary = mlb.transform(y_pred)
    return y_pred_binary, y_true_binary


class Metric:
    @classmethod
    def averaged(
        cls,
        y_pred: List[List],
        y_true: List[List],
        metric_types: List[str] = ["micro", "macro", "samples", "weighted"],
    ) -> Dict:
        y_pred_binary, y_true_binary = _transform(y_pred, y_true)
        resp = {}
        for metric_type in metric_types:
            assert metric_type in [
                "micro",
                "macro",
                "samples",
                "weighted",
            ], "metric type error."
            resp["{}-Averaged Precision".format(metric_type)] = precision_score(
                y_true_binary, y_pred_binary, average=metric_type
            )
            resp["{}-Averaged Recall".format(metric_type)] = recall_score(
                y_true_binary, y_pred_binary, average=metric_type
            )
            resp["{}-Averaged F1".format(metric_type)] = f1_score(
                y_true_binary, y_pred_binary, average=metric_type
            )
        return resp

    @classmethod
    def jaccard(
        cls,
        y_pred: List[List],
        y_true: List[List],
    ) -> Dict:
        def jaccard_similarity(l_pred: List, l_true: List) -> float:
            intersection = len(set(l_pred) & set(l_true))
            union = len(set(l_pred) | set(l_true))
            if union == 0:
                return 0
            else:
                return intersection / union

        similarities = [
            jaccard_similarity(l_pred, l_true) for l_pred, l_true in zip(y_pred, y_true)
        ]

        jaccard = sum(similarities) / len(similarities)
        return {"Jaccard Similarity": jaccard}

    @classmethod
    def hamming(
        cls,
        y_pred: List[List],
        y_true: List[List],
    ) -> Dict:
        y_pred_binary, y_true_binary = _transform(y_pred, y_true)
        hamming_loss = np.sum(y_true_binary != y_pred_binary) / (
            y_true_binary.shape[0] * y_true_binary.shape[1]
        )
        return {"Hamming Loss": hamming_loss}

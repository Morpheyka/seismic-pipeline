"""Structured channel quality adapter for binary/ordinal quality models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .hypno_features import MLPChannelQualityPredictor


CHANNEL_NAMES = {
    0: "cxf",
    1: "cxb",
    2: "ecg",
    3: "emg",
}


@dataclass(frozen=True)
class ChannelQualityDecision:
    """Model output for one channel with selection decision."""

    channel_index: int
    channel_name: str
    predicted_class: int
    probability_good: float
    is_selected: bool
    reason: str


class ChannelQualityAdapter:
    """Predict and explain quality decisions for requested channels."""

    def __init__(
        self,
        model_path: str,
        module_paths: Optional[List[str]] = None,
        good_classes: Tuple[int, ...] = (1,),
        rat_dir: Optional[str] = None,
        probability_threshold: float = 0.0,
    ) -> None:
        if len(good_classes) == 0:
            raise ValueError("good_classes must not be empty")
        self.model_path = str(model_path)
        self.module_paths = [str(Path(p)) for p in (module_paths or [])]
        self.good_classes = tuple(int(v) for v in good_classes)
        self.rat_dir = rat_dir
        self.probability_threshold = float(probability_threshold)
        self.predictor = MLPChannelQualityPredictor(
            model_path=self.model_path,
            module_paths=self.module_paths,
            good_classes=self.good_classes,
            rat_dir=self.rat_dir,
        )

    def evaluate(
        self,
        rat_id: str,
        date: str,
        channels: Sequence[int],
    ) -> Dict[int, ChannelQualityDecision]:
        """Return structured model decisions for each channel index."""
        date_norm = date.replace("-", "_")
        decisions: Dict[int, ChannelQualityDecision] = {}
        for ch in channels:
            ch_idx = int(ch)
            ch_name = CHANNEL_NAMES.get(ch_idx, f"ch{ch_idx}")
            pred_class, prob_good = self.predictor.predict_class_and_proba(
                rat_id=str(rat_id),
                date=date_norm,
                channel_1b=ch_idx + 1,
            )
            is_good_class = pred_class in self.good_classes
            passed_probability = prob_good >= self.probability_threshold
            is_selected = bool(is_good_class and passed_probability)
            if is_selected:
                reason = "selected"
            elif not is_good_class:
                reason = f"class_{pred_class}_not_in_good_classes"
            else:
                reason = f"probability_below_threshold_{self.probability_threshold:.2f}"
            decisions[ch_idx] = ChannelQualityDecision(
                channel_index=ch_idx,
                channel_name=ch_name,
                predicted_class=int(pred_class),
                probability_good=float(prob_good),
                is_selected=is_selected,
                reason=reason,
            )
        return decisions

    def to_rows(self, decisions: Dict[int, ChannelQualityDecision]) -> List[Dict[str, object]]:
        """Serialize decisions to notebook/log friendly dictionaries."""
        rows: List[Dict[str, object]] = []
        for ch in sorted(decisions):
            d = decisions[ch]
            rows.append(
                {
                    "channel_index": d.channel_index,
                    "channel_name": d.channel_name,
                    "predicted_class": d.predicted_class,
                    "probability_good": d.probability_good,
                    "is_selected": d.is_selected,
                    "reason": d.reason,
                }
            )
        return rows

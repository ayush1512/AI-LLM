from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class MetricResult:
    metric_name: str
    score: float  # Normalized 0.0 to 1.0
    passed: bool
    reason: str
    details: Optional[Dict[str, Any]] = None

class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        expected: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        pass


from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class RailResult:
    passed: bool
    rail_name: str
    action: str  # "allow", "block", "redact"
    reason: Optional[str] = None
    sanitized_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseRail(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass


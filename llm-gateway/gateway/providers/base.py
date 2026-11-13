from abc import ABC, abstractmethod
from typing import Dict, Any, List

class LLMProvider(ABC):
    """Abstract base class for all LLM Provider adapters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the provider credentials and endpoints are configured."""
        pass

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """
        Translates unified OpenAI-format request into provider format, executes,
        and translates response back into OpenAI chat completion schema.
        """
        pass

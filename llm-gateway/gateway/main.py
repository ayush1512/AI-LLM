from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from gateway.router import GatewayRouter
from config import config

try:
    from fastapi import FastAPI, Header, HTTPException
    fastapi_available = True
except ImportError:
    fastapi_available = False

if fastapi_available:
    app = FastAPI(
        title="OpenAI-Compatible LLM Gateway",
        description="Unified API Gateway proxying multiple LLM providers with caching, rate limiting, and fallback.",
        version="1.0.0"
    )
else:
    app = None

router = GatewayRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    provider: Optional[str] = None
    stream: Optional[bool] = False

if fastapi_available:
    @app.get("/health")
    def health():
        return {"status": "healthy", "default_provider": config.default_provider}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
                {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
                {"id": "claude-3-5-sonnet", "object": "model", "owned_by": "anthropic"},
                {"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
                {"id": "llama3", "object": "model", "owned_by": "ollama"},
                {"id": "mock-model", "object": "model", "owned_by": "mock"}
            ]
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        req: ChatCompletionRequest,
        authorization: Optional[str] = Header(None)
    ):
        api_key = ""
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "").strip()
        elif authorization:
            api_key = authorization.strip()
        else:
            api_key = config.master_key

        try:
            messages_dicts = [m.model_dump() for m in req.messages]
            result = router.route_completion(
                messages=messages_dicts,
                model=req.model,
                api_key=api_key,
                preferred_provider=req.provider,
                temperature=req.temperature or 0.7
            )
            return result
        except PermissionError as pe:
            raise HTTPException(status_code=401, detail=str(pe))
        except ValueError as ve:
            raise HTTPException(status_code=429, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

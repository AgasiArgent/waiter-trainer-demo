"""
Waiter Trainer Demo - FastAPI Application
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path

import httpx

from .llm import MistralClient, run_training

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Concurrency limiter — prevent API quota exhaustion
MAX_CONCURRENT_TRAININGS = 2
training_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRAININGS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup/shutdown hooks."""
    # Startup
    if not os.getenv("MISTRAL_API_KEY", "").strip():
        raise RuntimeError("MISTRAL_API_KEY is not set — cannot start")
    app.state.mistral_client = MistralClient()
    logger.info("Application started, API key configured")
    yield
    # Shutdown
    await app.state.mistral_client.close()
    logger.info("Application shutting down")


app = FastAPI(title="Waiter Trainer Demo", lifespan=lifespan)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Mount static files if directory exists
static_dir = BASE_DIR.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class TrainingRequest(BaseModel):
    guest_type: Literal["friendly", "couple", "wine"]
    waiter_level: Literal["novice", "experienced", "expert"]


class DialogMessage(BaseModel):
    role: Literal["guest", "waiter"]
    content: str


class TrainingResponse(BaseModel):
    messages: list[DialogMessage]
    evaluation: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/training", response_model=TrainingResponse)
async def training(req: TrainingRequest, request: Request):
    """Generate training dialog and evaluation."""
    # Acquire semaphore with asyncio.timeout (safe on Python 3.11+, avoids wait_for race)
    try:
        async with asyncio.timeout(1.0):
            await training_semaphore.acquire()
    except TimeoutError:
        raise HTTPException(429, "Too many concurrent requests. Please try again later.")

    try:
        result = await asyncio.wait_for(
            run_training(req.guest_type, req.waiter_level, request.app.state.mistral_client),
            timeout=300  # 5 minutes max
        )
        return TrainingResponse(**result)
    except asyncio.TimeoutError:
        logger.error("Training timed out for %s/%s", req.guest_type, req.waiter_level)
        raise HTTPException(500, "Training generation timed out")
    except httpx.HTTPStatusError as e:
        logger.exception("Mistral API HTTP error: %s", e.response.status_code)
        raise HTTPException(502, "Internal server error")
    except FileNotFoundError:
        logger.exception("Prompt file not found")
        raise HTTPException(500, "Internal server error")
    except RuntimeError:
        logger.exception("API response error")
        raise HTTPException(500, "Internal server error")
    except ValueError:
        logger.exception("Configuration error")
        raise HTTPException(500, "Internal server error")
    except Exception:
        logger.exception("Training generation failed")
        raise HTTPException(500, "Internal server error")
    finally:
        training_semaphore.release()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

"""
Mistral API integration for waiter trainer.
"""
import asyncio
import logging
import os
import re
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"

# Rate limiting settings
MAX_RETRIES = 5
INITIAL_DELAY = 2.0  # seconds
MAX_DELAY = 30.0  # seconds

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Dialog termination phrases
WAITER_DONE_PHRASES = ["принесу", "сейчас оформлю", "записал заказ"]
GUEST_DONE_PHRASES = [
    "возьму", "возьмём", "беру", "берём", "заказываю", "закажем",
    "давайте это", "это всё",
    "до свидания", "до встречи", "было приятно", "приятного вечера"
]


def _normalize_yo(text: str) -> str:
    """Normalize ё -> е (LLMs often output е instead of ё)."""
    return text.replace('ё', 'е').replace('Ё', 'Е')


def _contains_phrase(text: str, phrases: list[str]) -> bool:
    """Check if text contains any phrase as a whole word (word-boundary aware)."""
    lower = _normalize_yo(text.lower())
    for phrase in phrases:
        if re.search(r'\b' + re.escape(_normalize_yo(phrase)) + r'\b', lower):
            return True
    return False


def is_waiter_done(reply: str) -> bool:
    """Check if waiter's reply signals dialog completion."""
    return _contains_phrase(reply, WAITER_DONE_PHRASES)


def is_guest_done(reply: str) -> bool:
    """Check if guest's reply signals dialog completion."""
    return _contains_phrase(reply, GUEST_DONE_PHRASES)


def load_prompt(filename: str) -> str:
    """Load prompt from file (path-safe)."""
    path = (PROMPTS_DIR / filename).resolve()
    if not path.is_relative_to(PROMPTS_DIR.resolve()):
        raise ValueError(f"Invalid prompt filename: {filename}")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {filename}")
    return path.read_text(encoding="utf-8")


class MistralClient:
    """Async client for Mistral API with connection reuse."""

    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Send chat request with retry logic for rate limits."""
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        delay = INITIAL_DELAY
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.post(
                    MISTRAL_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": MISTRAL_MODEL,
                        "messages": api_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    await response.aclose()
                    wait = delay
                    if retry_after:
                        try:
                            wait = min(float(retry_after), MAX_DELAY)
                        except ValueError:
                            pass  # Non-numeric Retry-After (HTTP-date), use backoff delay
                    logger.warning(
                        "Rate limited. Waiting %.1fs (attempt %d/%d)",
                        wait, attempt + 1, MAX_RETRIES
                    )
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, MAX_DELAY)
                    last_error = RuntimeError(
                        f"Rate limited after {attempt + 1} attempts"
                    )
                    continue

                response.raise_for_status()
                data = response.json()

                # Defensive response parsing
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("Empty choices in API response")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise RuntimeError("Empty content in API response")
                return content

            except httpx.HTTPStatusError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                logger.warning(
                    "Transport error: %s (attempt %d/%d)",
                    e, attempt + 1, MAX_RETRIES
                )
                last_error = e
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)
                continue

        raise last_error or RuntimeError("Max retries exceeded")


async def run_training(guest_type: str, waiter_level: str, client: MistralClient) -> dict:
    """
    Generate full training dialog + evaluation.

    Args:
        guest_type: 'friendly', 'couple', or 'wine'
        waiter_level: 'novice', 'experienced', or 'expert'
        client: MistralClient instance (app-scoped)

    Returns:
        {
            "messages": [{"role": "guest"|"waiter", "content": "..."}],
            "evaluation": "markdown evaluation"
        }
    """
    # Load prompts
    guest_prompt = load_prompt(f"guest-{guest_type}.md")
    waiter_prompt = load_prompt(f"waiter-{waiter_level}.md")
    menu_data = load_prompt("menu-data.md")
    evaluator_prompt = load_prompt("evaluator.md")

    # Combined system prompts
    guest_system = f"{guest_prompt}\n\n## ДАННЫЕ МЕНЮ\n\n{menu_data}"
    waiter_system = f"{waiter_prompt}\n\n## ДАННЫЕ МЕНЮ\n\n{menu_data}"
    evaluator_system = f"{evaluator_prompt}\n\n## ДАННЫЕ МЕНЮ\n\n{menu_data}"

    # Generate dialog
    messages = []
    guest_history = []
    waiter_history = []

    # Start with guest's opening line
    guest_reply = await client.chat(
        guest_system,
        [{"role": "user", "content": "Начни диалог. Ты только что сел за стол."}],
        temperature=0.8
    )
    messages.append({"role": "guest", "content": guest_reply})
    guest_history.append({"role": "assistant", "content": guest_reply})
    waiter_history.append({"role": "user", "content": guest_reply})

    # Generate dialog turns (up to 12 exchanges)
    for turn in range(12):
        # Waiter responds
        waiter_reply = await client.chat(
            waiter_system,
            waiter_history,
            temperature=0.7
        )
        messages.append({"role": "waiter", "content": waiter_reply})
        waiter_history.append({"role": "assistant", "content": waiter_reply})
        guest_history.append({"role": "user", "content": waiter_reply})

        # Check for dialog completion signals from waiter
        if is_waiter_done(waiter_reply):
            break

        # Guest responds
        guest_reply = await client.chat(
            guest_system,
            guest_history,
            temperature=0.8
        )
        messages.append({"role": "guest", "content": guest_reply})
        guest_history.append({"role": "assistant", "content": guest_reply})
        waiter_history.append({"role": "user", "content": guest_reply})

        # Check for order completion or farewell from guest
        if is_guest_done(guest_reply):
            # One more waiter response to close
            waiter_final = await client.chat(
                waiter_system,
                waiter_history,
                temperature=0.7
            )
            messages.append({"role": "waiter", "content": waiter_final})
            break
    else:
        # Loop exhausted without break — add waiter closing
        logger.warning("Dialog loop exhausted 12 turns without completion signal")
        waiter_final = await client.chat(
            waiter_system,
            waiter_history,
            temperature=0.7
        )
        messages.append({"role": "waiter", "content": waiter_final})

    logger.info("Dialog completed: %d messages", len(messages))

    # Generate evaluation
    transcript = "\n".join([
        f"{'Гость' if m['role'] == 'guest' else 'Официант'}: {m['content']}"
        for m in messages
    ])

    evaluation = await client.chat(
        evaluator_system,
        [{"role": "user", "content": f"Оцени следующий диалог:\n\n{transcript}"}],
        temperature=0.3,
        max_tokens=16384
    )

    logger.info("Evaluation generated: %d chars", len(evaluation))

    return {
        "messages": messages,
        "evaluation": evaluation
    }

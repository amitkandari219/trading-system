"""
Configurable LLM client for signal extraction.
Supports:
  - Ollama (free, local) — default
  - Anthropic Claude API (requires ANTHROPIC_API_KEY)

Set LLM_BACKEND env var to choose:
  LLM_BACKEND=ollama   (default, free)
  LLM_BACKEND=anthropic

For Ollama, install and run:
  brew install ollama
  ollama pull llama3.1:8b      # for extraction (fast)
  ollama pull llama3.1:70b     # for hallucination check (if you have RAM)
  ollama serve
"""

import json
import os
import re
import logging

logger = logging.getLogger(__name__)

LLM_BACKEND = os.environ.get('LLM_BACKEND', 'anthropic')

# Ollama models — change these if you prefer different models
OLLAMA_EXTRACTION_MODEL = os.environ.get('OLLAMA_EXTRACTION_MODEL', 'llama3.2:latest')
OLLAMA_HALLUCINATION_MODEL = os.environ.get('OLLAMA_HALLUCINATION_MODEL', 'llama3.2:latest')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

# Anthropic models (only used if LLM_BACKEND=anthropic)
ANTHROPIC_EXTRACTION_MODEL = 'claude-haiku-4-5-20251001'
ANTHROPIC_HALLUCINATION_MODEL = 'claude-sonnet-4-20250514'


class LLMClient:
    """
    Unified LLM client that routes to Ollama or Anthropic.
    All methods return parsed JSON dicts or None on failure.
    """

    def __init__(self, backend=None):
        self.backend = backend or LLM_BACKEND

        if self.backend == 'anthropic':
            import anthropic
            self.anthropic_client = anthropic.Anthropic()
        elif self.backend == 'ollama':
            import requests
            self._requests = requests
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def extract(self, prompt: str) -> dict:
        """Call extraction model (fast, cheap)."""
        if self.backend == 'anthropic':
            return self._call_anthropic(prompt, ANTHROPIC_EXTRACTION_MODEL)
        else:
            return self._call_ollama(prompt, OLLAMA_EXTRACTION_MODEL)

    def hallucination_check(self, prompt: str) -> dict:
        """Call hallucination check model (more careful)."""
        if self.backend == 'anthropic':
            return self._call_anthropic(prompt, ANTHROPIC_HALLUCINATION_MODEL)
        else:
            return self._call_ollama(prompt, OLLAMA_HALLUCINATION_MODEL)

    def _call_anthropic(self, prompt: str, model: str) -> dict:
        """Call Anthropic Claude API."""
        try:
            message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw = message.content[0].text.strip()
            return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Anthropic API error: {e}")
            return None

    def _call_ollama(self, prompt: str, model: str) -> dict:
        """Call Ollama local API."""
        try:
            response = self._requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 1500,
                    },
                    'format': 'json',
                },
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json().get('response', '').strip()
            return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Ollama API error: {e}")
            return None

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from LLM response, stripping markdown fences and trailing text."""
        try:
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            # Try direct parse first
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Extract first JSON object from response
        try:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        logger.warning(f"JSON parse error. Raw: {raw[:200]}")
        return None

    @property
    def backend_name(self) -> str:
        if self.backend == 'ollama':
            return f"Ollama ({OLLAMA_EXTRACTION_MODEL})"
        return f"Anthropic ({ANTHROPIC_EXTRACTION_MODEL})"

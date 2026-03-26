"""
model_manager.py
----------------
Handles loading and switching between fine-tuned LoRA adapters:
  - Qwen3-4B        (qwen-model-ok)
  - SeaLLMs-v3-7B-Chat (SeaLLM-model-ok)

KEY DESIGN: device_map is NEVER passed to AutoModelForCausalLM.
  Any device_map value triggers accelerate's meta-device sharding —
  layers are created as empty "meta" tensors with no real data.
  PeftModel.from_pretrained() then crashes trying to copy them:
    "Cannot copy out of meta tensor; no data!"
  Without device_map, bitsandbytes handles CUDA placement itself
  during the quantization pass → real tensors, no crash.
"""

import gc
import os
import torch
import logging
import threading
from pathlib import Path
from threading import Lock
from typing import Optional, Generator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars only

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter registry  ←  change MODEL_ROOT if you move the adapter folders
# ---------------------------------------------------------------------------
# Support both local and cloud environments via environment variable
# Set MODEL_ROOT_PATH env var to override default location
# Example: export MODEL_ROOT_PATH=/teamspace/studios/this_studio/model-for-inference
MODEL_ROOT = Path(os.getenv("MODEL_ROOT_PATH", Path(__file__).resolve().parent.parent))
logger.info(f"MODEL_ROOT set to: {MODEL_ROOT}")

ADAPTERS = {
    "qwen": {
        "label": "Qwen3-4B (QLoRA fine-tuned)",
        "base_model": "unsloth/qwen3-4b-unsloth-bnb-4bit",
        "adapter_path": "/teamspace/studios/this_studio/model",
        "is_instruct": True,
        "emoji": "🟦",
    },
    "seallm": {
        "label": "SeaLLMs-v3-7B-Chat (QLoRA fine-tuned)",
        "base_model": "SeaLLMs/SeaLLMs-v3-7B-Chat",
        "adapter_path": "/teamspace/studios/this_studio/seallm/Check-point_4500",
        "is_instruct": True,
        "emoji": "🟩",
    },
}


class ModelManager:
    """Thread-safe singleton that loads one model at a time."""

    def __init__(self):
        self._lock = Lock()
        self.current_model_id: Optional[str] = None
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, quantization: str = "4bit") -> None:
        """Load the requested adapter, unloading the previous one first."""
        with self._lock:
            if self.current_model_id == model_id:
                logger.info(f"Model '{model_id}' is already loaded.")
                return

            self._unload()
            cfg = ADAPTERS[model_id]
            logger.info(f"Loading {cfg['label']} ...")

            # ── GPU info ──────────────────────────────────────────────
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                logger.info(
                    f"GPU: {torch.cuda.get_device_name(0)} "
                    f"| Free VRAM: {free_gb:.1f} GB"
                )
            else:
                logger.warning("No CUDA GPU — running on CPU (very slow).")

            # ── BitsAndBytes quantization config ──────────────────────
            bnb_config = self._make_bnb_config(quantization)

            # ── Load base model ───────────────────────────────────────
            #
            # *** CRITICAL: do NOT pass device_map here ***
            #
            # Passing any device_map (including "auto" or {"": 0}) causes
            # accelerate to shard the model using "meta" tensors — placeholder
            # objects with shape/dtype but ZERO bytes of real data.
            # PeftModel then tries to call .to() on those tensors → crash:
            #   "Cannot copy out of meta tensor; no data!"
            #
            # Without device_map, bitsandbytes places real quantized tensors
            # directly on CUDA during from_pretrained() — no meta device,
            # no crash, PeftModel wraps it cleanly.
            #
            logger.info(f"Loading base weights: {cfg['base_model']}")
            base = AutoModelForCausalLM.from_pretrained(
                cfg["base_model"],
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

            # ── Wrap with LoRA adapter ────────────────────────────────
            logger.info(f"Merging LoRA adapter: {cfg['adapter_path']}")
            merged = PeftModel.from_pretrained(
                base,
                cfg["adapter_path"],
                is_trainable=False,
            )
            merged.eval()

            # ── Tokenizer (must match adapter's vocab) ─────────────────
            tokenizer = AutoTokenizer.from_pretrained(
                cfg["adapter_path"],
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.model = merged
            self.tokenizer = tokenizer
            self.current_model_id = model_id
            device_str = str(next(merged.parameters()).device)
            logger.info(f"✅ '{cfg['label']}' ready on {device_str}")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        """Generate a response, optionally streaming tokens."""
        with self._lock:
            if self.model is None:
                raise RuntimeError("No model loaded. Call load_model() first.")

            prompt = self._build_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
                self.model.device
            )

            gen_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            if stream:
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                gen_kwargs["streamer"] = streamer

                t = threading.Thread(
                    target=self.model.generate, kwargs=gen_kwargs, daemon=True
                )
                t.start()

                for token in streamer:
                    yield token
            else:
                with torch.no_grad():
                    output = self.model.generate(**gen_kwargs)
                input_len = inputs["input_ids"].shape[1]
                answer_ids = output[0][input_len:]
                yield self.tokenizer.decode(answer_ids, skip_special_tokens=True)

    def get_status(self) -> dict:
        return {
            "loaded": self.current_model_id is not None,
            "model_id": self.current_model_id,
            "label": ADAPTERS[self.current_model_id]["label"]
            if self.current_model_id
            else None,
            "device": str(next(self.model.parameters()).device) if self.model else None,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unload(self):
        if self.model is not None:
            logger.info(f"Unloading '{self.current_model_id}' and freeing VRAM …")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _make_bnb_config(quantization: str) -> Optional[BitsAndBytesConfig]:
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None  # full fp32/bf16

    def _build_prompt(self, messages: list[dict]) -> str:
        """Use the tokenizer's chat template if available, else manual ChatML format."""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # For Qwen models: add empty think block to skip thinking and answer directly
            # This tells the model "thinking is done, start answering"
            if self.current_model_id == "qwen":
                prompt = prompt + "<think>\n\n</think>\n\n"
            return prompt
        except Exception:
            # Fallback for adapters without a bundled chat template (ChatML format)
            parts = []
            for m in messages:
                role, content = m["role"], m["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            # Add empty think block for Qwen
            if self.current_model_id == "qwen":
                parts.append("<think>\n\n</think>\n\n")
            return "\n".join(parts)


# Global singleton
manager = ModelManager()

import gc
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter registry - Absolute paths from your screenshots
# ---------------------------------------------------------------------------
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

    def load_model(self, model_id: str, quantization: str = "4bit") -> None:
        """Load the requested adapter, unloading the previous one first."""
        with self._lock:
            if self.current_model_id == model_id:
                logger.info(f"Model '{model_id}' is already loaded.")
                return

            self._unload()
            cfg = ADAPTERS[model_id]
            logger.info(f"Loading {cfg['label']} ...")

            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                logger.info(f"GPU: {torch.cuda.get_device_name(0)} | Free VRAM: {free_gb:.1f} GB")

            bnb_config = self._make_bnb_config(quantization)

            # ── Load base model ──
            logger.info(f"Loading base weights: {cfg['base_model']}")
            base = AutoModelForCausalLM.from_pretrained(
                cfg["base_model"],
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

            # ── Wrap with LoRA adapter ──
            logger.info(f"Merging LoRA adapter: {cfg['adapter_path']}")
            merged = PeftModel.from_pretrained(
                base,
                cfg["adapter_path"],
                is_trainable=False,
            )
            merged.eval()

            # ── Tokenizer (FIX: chat_template=None stops the crash during load) ──
            logger.info(f"Loading tokenizer from: {cfg['adapter_path']}")
            tokenizer = AutoTokenizer.from_pretrained(
                cfg["adapter_path"],
                trust_remote_code=True,
                chat_template=None  # <--- THIS IS THE CRITICAL FIX
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
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True
            ).to(self.model.device)

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
                t = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
                t.start()
                for token in streamer:
                    yield token
            else:
                with torch.no_grad():
                    output = self.model.generate(**gen_kwargs)
                input_len = inputs["input_ids"].shape[1]
                answer_ids = output[0][input_len:]
                yield self.tokenizer.decode(answer_ids, skip_special_tokens=True)

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
        return BitsAndBytesConfig(load_in_8bit=True) if quantization == "8bit" else None

    def _build_prompt(self, messages: list[dict]) -> str:
        """Manual ChatML construction to bypass the broken template entirely."""
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

# Global singleton
manager = ModelManager()
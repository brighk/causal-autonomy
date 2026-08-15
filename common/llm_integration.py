"""
LLM Integration for CAF Experiments
====================================
Supports:
- Hugging Face local models
- Ollama-served local models such as gemma4:e4b
- Constraint injection into prompts
- Configurable generation parameters
"""

import os
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib import request as urllib_request
from urllib import error as urllib_error

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except Exception:  # pragma: no cover - exercised indirectly through runtime selection
    torch = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
        logging as transformers_logging,
    )
except Exception:  # pragma: no cover - exercised indirectly through runtime selection
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    pipeline = None
    transformers_logging = None

# Suppress the max_new_tokens/max_length warning spam
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
if transformers_logging is not None:
    transformers_logging.set_verbosity_error()

from experiments.caf_algorithm import InferenceLayer


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _strip_think_block(text: str) -> str:
    """Remove reasoning-model <think>...</think> blocks from generated text.

    Reasoning models such as Qwen3 emit their chain of thought inside
    <think> tags before the actual answer. Downstream proposition parsing
    must only see the answer. If the block is unterminated (generation
    truncated mid-thought), everything from <think> onward is dropped.
    """
    if "<think>" not in text:
        return text
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in stripped:
        stripped = stripped.split("<think>", 1)[0]
    return stripped


@dataclass
class LLMConfig:
    """Configuration for local LLM backends."""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"  # or Llama-3-8b-Instruct
    device: str = "cuda"  # Use GPU
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    load_in_8bit: bool = False  # Set True for memory efficiency
    load_in_4bit: bool = False  # Set True for even more efficiency
    trust_remote_code: bool = False
    backend: str = "hf"
    ollama_base_url: str = "http://localhost:11434"
    request_timeout: int = 300
    ollama_num_ctx: Optional[int] = None
    ollama_num_thread: Optional[int] = None


class HuggingFaceCausalLMLayer(InferenceLayer):
    """
    Real Inference Layer using any Hugging Face causal LM.

    Loads a model locally on GPU (via AutoModelForCausalLM) and generates
    responses with optional constraint injection. Prompt formatting adapts
    to the model family (Llama, Qwen, or a generic fallback).

    Example models:
    - meta-llama/Llama-2-7b-chat-hf (requires HF token, gated)
    - meta-llama/Llama-3-8B-Instruct (newer, better)
    - meta-llama/Llama-2-13b-chat-hf (larger, more capable)
    - NousResearch/Llama-2-7b-chat-hf (ungated alternative)
    - Qwen/Qwen3-14B, Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None:
            raise RuntimeError(
                "Hugging Face inference was requested, but required dependencies are missing. "
                "Install torch and transformers, or use an Ollama-backed model such as 'gemma4:e4b'."
            )
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.config.device}")

        # Check GPU availability
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.config.device = "cpu"

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Configure quantization for memory efficiency
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
            # On small GPUs (e.g., 4GB), keep headroom and offload overflow to CPU.
            if self.config.device == "cuda" and torch.cuda.is_available():
                total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
                total_mem_gib = max(1, int(total_mem_bytes / (1024 ** 3)))
                gpu_budget_gib = max(1, total_mem_gib - 1)
                model_kwargs["max_memory"] = {
                    0: f"{gpu_budget_gib}GiB",
                    "cpu": "48GiB"
                }
                offload_dir = Path(".hf_offload")
                offload_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs["offload_folder"] = str(offload_dir)
                model_kwargs["offload_state_dict"] = True
        else:
            model_kwargs["device_map"] = self.config.device

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_new_tokens = self.config.max_new_tokens

        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if quantization_config else self.config.device,
        )

        print(f"Model loaded successfully on {self.config.device}")

    def _format_prompt(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Format prompt for the model's chat template, with optional constraints.

        Dispatches to the right template by model family (Llama 2/3, Qwen,
        or a generic instruction-style fallback for anything else).
        """
        system_message = """You are a helpful AI assistant that provides accurate, logically consistent responses grounded in factual knowledge.
Your responses should be clear, well-reasoned, and avoid contradictions."""

        # Add constraints if provided
        if constraints and len(constraints) > 0:
            system_message += "\n\nIMPORTANT CONSTRAINTS to follow:"
            for i, constraint in enumerate(constraints, 1):
                system_message += f"\n{i}. {constraint}"

        # Format using Llama chat template
        # For Llama 2
        if "Llama-2" in self.config.model_name:
            formatted_prompt = f"""[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]"""
        # For Llama 3 (uses different format)
        elif "Llama-3" in self.config.model_name or "llama-3" in self.config.model_name.lower():
            # Llama 3 uses a different chat template
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif "qwen" in self.config.model_name.lower():
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            # enable_thinking=False disables Qwen3's <think> reasoning blocks,
            # which would otherwise consume the token budget and break
            # proposition parsing. Qwen2.5 templates ignore the flag.
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            # Fallback for other models: avoid chat-loop style prompting.
            formatted_prompt = (
                f"{system_message}\n\n"
                f"Task:\n{prompt}\n\n"
                "Respond concisely and directly."
            )

        return formatted_prompt

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response using the loaded model.

        Args:
            prompt: Input prompt
            constraints: Optional list of constraints to inject

        Returns:
            Generated response text
        """
        formatted_prompt = self._format_prompt(prompt, constraints)

        # Generate
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_return_sequences=1,
            return_full_text=False,  # Only return generated text
        )

        # Extract generated text
        response = outputs[0]["generated_text"]

        # Clean up response (remove any residual prompt artifacts)
        response = _strip_think_block(response).strip()

        return response

    def generate_batch(
        self,
        prompts: List[str],
        constraints: Optional[List[List[str]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        formatted_prompts: List[str] = []
        for idx, prompt in enumerate(prompts):
            per_constraints = constraints[idx] if constraints and idx < len(constraints) else None
            formatted_prompts.append(self._format_prompt(prompt, per_constraints))

        outputs = self.pipeline(
            formatted_prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_return_sequences=1,
            return_full_text=False,
            batch_size=batch_size,
        )

        responses: List[str] = []
        for output in outputs:
            if isinstance(output, list):
                if output:
                    responses.append(_strip_think_block(str(output[0].get("generated_text", ""))).strip())
                else:
                    responses.append("")
            else:
                responses.append(_strip_think_block(str(output.get("generated_text", ""))).strip())
        return responses

    def __del__(self):
        """Clean up GPU memory when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class OllamaLayer(InferenceLayer):
    """Inference layer backed by a locally running Ollama server."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig(
            model_name="gemma4:e4b",
            backend="ollama",
            device="cpu",
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

    def _format_prompt(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None,
    ) -> str:
        system_message = (
            "You are a helpful AI assistant that provides accurate, logically consistent "
            "responses grounded in factual knowledge."
        )
        if constraints:
            system_message += "\n\nIMPORTANT CONSTRAINTS:"
            for idx, constraint in enumerate(constraints, start=1):
                system_message += f"\n{idx}. {constraint}"
        return f"{system_message}\n\nTask:\n{prompt}"

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": self.config.model_name,
            "prompt": self._format_prompt(prompt, constraints),
            "stream": False,
            "options": {
                "num_predict": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
        }
        if self.config.ollama_num_ctx is not None:
            payload["options"]["num_ctx"] = self.config.ollama_num_ctx
        if self.config.ollama_num_thread is not None:
            payload["options"]["num_thread"] = self.config.ollama_num_thread
        req = urllib_request.Request(
            f"{self.config.ollama_base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.config.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw)
                detail = str(parsed.get("error", raw)).strip()
            except Exception:
                detail = str(exc).strip()

            if "requires more system memory" in detail:
                raise RuntimeError(
                    "Ollama could not load the requested model because the machine does not "
                    f"have enough free RAM right now. Server message: {detail}"
                ) from exc

            raise RuntimeError(f"Ollama request failed: {detail or exc}") from exc
        return str(data.get("response", "")).strip()

    def generate_batch(
        self,
        prompts: List[str],
        constraints: Optional[List[List[str]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        results: List[str] = []
        for idx, prompt in enumerate(prompts):
            per_constraints = constraints[idx] if constraints and idx < len(constraints) else None
            results.append(self.generate(prompt, per_constraints))
        return results


class OpenSourceLlamaLayer(InferenceLayer):
    """
    Alternative implementation using ungated open-source Llama variants.

    Uses models that don't require Meta's gated access:
    - NousResearch/Llama-2-7b-chat-hf
    - NousResearch/Llama-2-13b-chat-hf
    - teknium/OpenHermes-2.5-Mistral-7B (Mistral-based alternative)
    """

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-chat-hf",
        device: str = "cuda",
        load_in_4bit: bool = False
    ):
        config = LLMConfig(
            model_name=model_name,
            device=device,
            load_in_4bit=load_in_4bit
        )
        self.hf_layer = HuggingFaceCausalLMLayer(config)

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        return self.hf_layer.generate(prompt, constraints)

    def generate_batch(
        self,
        prompts: List[str],
        constraints: Optional[List[List[str]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        return self.hf_layer.generate_batch(prompts, constraints, batch_size)


# Convenience factory function
def create_causal_lm_layer(
    model_size: str = "7b",
    use_4bit: bool = False,
    use_8bit: bool = False,
    open_source: bool = True
) -> InferenceLayer:
    """
    Factory function to create an inference layer (HuggingFace or Ollama backend).

    Args:
        model_size: Model size/alias, a direct HF path, or an "ollama:<tag>" name:
            - "7b": Llama-2-7B (3.5GB with 4-bit)
            - "8b": Llama-3-8B (4GB with 4-bit)
            - "13b": Llama-2-13B (7GB with 4-bit - needs 8GB+ GPU)
            - "tiny": TinyLlama-1.1B (0.6GB with 4-bit - ultra-lightweight)
            - "phi2": Microsoft Phi-2 2.7B (1.5GB with 4-bit - good balance)
            - "mistral": Mistral-7B (3.5GB with 4-bit - Llama alternative)
            - "qwen7b"/"qwen14b": Qwen2.5-Instruct aliases
            - any "org/model" HF path (e.g. "Qwen/Qwen3-14B") loads directly
            - "ollama:<tag>" (e.g. "ollama:gemma4:e4b") routes to a running Ollama server
        use_4bit: Use 4-bit quantization (saves memory, recommended for 4GB GPU)
        use_8bit: Use 8-bit quantization
        open_source: Use ungated open-source variants (recommended)

    Returns:
        Configured HuggingFaceCausalLMLayer or OllamaLayer instance

    Examples:
        # For 4GB GPU - recommended
        llm = create_causal_lm_layer("7b", use_4bit=True)  # 3.5GB

        # For 4GB GPU - ultra-lightweight
        llm = create_causal_lm_layer("tiny", use_4bit=True)  # 0.6GB

        # For 4GB GPU - good balance
        llm = create_causal_lm_layer("phi2", use_4bit=True)  # 1.5GB

        # Direct HF path on a larger GPU
        llm = create_causal_lm_layer("Qwen/Qwen3-14B", use_4bit=True)  # ~9GB
    """
    if model_size.startswith("ollama:"):
        model_tag = model_size.split(":", 1)[1]
        return OllamaLayer(
            LLMConfig(
                model_name=model_tag,
                backend="ollama",
                device="cpu",
                max_new_tokens=512,
                temperature=0.0 if not use_8bit and not use_4bit else 0.0,
                top_p=1.0,
                do_sample=False,
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                request_timeout=_env_int("OLLAMA_REQUEST_TIMEOUT", 300) or 300,
                ollama_num_ctx=_env_int("OLLAMA_NUM_CTX"),
                ollama_num_thread=_env_int("OLLAMA_NUM_THREAD"),
            )
        )

    # Treat unknown colon-tagged models as Ollama model names, e.g. gemma4:e4b.
    known_hf_aliases = {"tiny", "phi2", "mistral", "qwen7b", "qwen14b", "qwen3-14b", "7b", "8b", "13b"}
    if ":" in model_size and model_size not in known_hf_aliases:
        return OllamaLayer(
            LLMConfig(
                model_name=model_size,
                backend="ollama",
                device="cpu",
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                request_timeout=_env_int("OLLAMA_REQUEST_TIMEOUT", 300) or 300,
                ollama_num_ctx=_env_int("OLLAMA_NUM_CTX"),
                ollama_num_thread=_env_int("OLLAMA_NUM_THREAD"),
            )
        )

    # Rough VRAM guidance for this code path (4-bit + runtime overhead).
    # GTX 1650 4GB typically cannot load phi2 reliably.
    min_vram_gib = {
        "tiny": 2,
        "phi2": 5,
        "mistral": 7,
        "qwen7b": 7,
        "qwen14b": 14,
        "qwen3-14b": 14,
        "7b": 7,
        "8b": 8,
        "13b": 12,
    }

    direct_model_name = "/" in model_size
    normalized_model_size = model_size.lower()
    required = min_vram_gib.get(model_size, 6)
    if direct_model_name:
        if "14b" in normalized_model_size:
            required = 14
        elif any(tag in normalized_model_size for tag in ("7b", "8b")):
            required = 8
        elif "13b" in normalized_model_size:
            required = 12

    if torch is not None and torch.cuda.is_available():
        total_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_mem_gib < required:
            allow_fallback = os.getenv("CAF_ALLOW_MODEL_FALLBACK", "0") == "1"
            if allow_fallback and model_size != "tiny":
                print(
                    f"WARNING: GPU has {total_mem_gib:.2f} GiB, but model '{model_size}' "
                    f"typically needs ~{required} GiB. Falling back to 'tiny' "
                    f"(set CAF_ALLOW_MODEL_FALLBACK=0 to disable)."
                )
                model_size = "tiny"
            elif model_size != "tiny":
                raise RuntimeError(
                    f"Insufficient VRAM for '{model_size}' on this GPU "
                    f"({total_mem_gib:.2f} GiB available, ~{required} GiB recommended).\n"
                    "Use '--llm-model tiny' or set CAF_ALLOW_MODEL_FALLBACK=1."
                )

    # Model map with small models for 4GB GPU support
    if direct_model_name:
        model_name = model_size
    else:
        if open_source:
            # Use ungated open-source models
            model_map = {
                "7b": "NousResearch/Llama-2-7b-chat-hf",
                "13b": "NousResearch/Llama-2-13b-chat-hf",
                # Small models for 4GB GPU
                "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "phi2": "microsoft/phi-2",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
                "qwen3-14b": "Qwen/Qwen3-14B",
            }
            model_name = model_map.get(model_size, model_map["7b"])
        else:
            # Use official Meta models (requires HF token and access)
            model_map = {
                "7b": "meta-llama/Llama-2-7b-chat-hf",
                "8b": "meta-llama/Llama-3-8B-Instruct",
                "13b": "meta-llama/Llama-2-13b-chat-hf",
                # Small models also available
                "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "phi2": "microsoft/phi-2",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
                "qwen3-14b": "Qwen/Qwen3-14B",
            }
            model_name = model_map.get(model_size, model_map["7b"])

    config = LLMConfig(
        model_name=model_name,
        device="cuda",
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
        backend="hf",
        trust_remote_code="qwen" in model_name.lower(),
    )

    return HuggingFaceCausalLMLayer(config)


if __name__ == "__main__":
    """Demo of LLM integration."""
    print("=" * 60)
    print("Testing LLM Integration")
    print("=" * 60)

    # Create inference layer (using 4-bit for memory efficiency)
    print("\nInitializing model...")
    llm = create_causal_lm_layer(model_size="7b", use_4bit=True, open_source=True)

    # Test basic generation
    print("\n" + "=" * 60)
    print("Test 1: Basic Generation")
    print("=" * 60)
    prompt1 = "Explain why rain makes roads slippery in 2-3 sentences."
    response1 = llm.generate(prompt1)
    print(f"Prompt: {prompt1}")
    print(f"Response: {response1}")

    # Test generation with constraints
    print("\n" + "=" * 60)
    print("Test 2: Generation with Constraints")
    print("=" * 60)
    prompt2 = "Explain how photosynthesis works."
    constraints = [
        "Focus on the chemical equation",
        "Mention chlorophyll",
        "Keep response under 3 sentences"
    ]
    response2 = llm.generate(prompt2, constraints)
    print(f"Prompt: {prompt2}")
    print(f"Constraints: {constraints}")
    print(f"Response: {response2}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

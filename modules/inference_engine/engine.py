"""
Module A: Inference Engine (Neural)
Hardware: NVIDIA A100/H100 GPU
Model: Llama-3-70B
Framework: PyTorch + vLLM for high-throughput inference
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

from common.llm_integration import HuggingFaceCausalLMLayer, LLMConfig

# Optional vLLM import
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available, will use Hugging Face transformers")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


@dataclass
class GenerationConfig:
    """Configuration for text generation"""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class InferenceEngine:
    """
    Neural inference engine for generating semantic hypotheses.

    Uses vLLM for optimized inference with PagedAttention and continuous batching.
    Generates:
    1. Natural language responses
    2. Causal assertions that can be verified
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-70b-chat-hf",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        use_vllm: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.model_name = model_name
        self.use_vllm = use_vllm and VLLM_AVAILABLE

        logger.info(f"Initializing Inference Engine with {model_name}")

        if self.use_vllm:
            self._init_vllm(tensor_parallel_size, gpu_memory_utilization)
        else:
            logger.info(
                "Using common.llm_integration.HuggingFaceCausalLMLayer (vLLM not available or disabled)"
            )
            self._init_huggingface(load_in_4bit, load_in_8bit)

        logger.info("Inference Engine initialized successfully")

    def _init_vllm(self, tensor_parallel_size: int, gpu_memory_utilization: float):
        """Initialize vLLM engine for high-performance inference"""
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=4096,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM engine initialized")

    def _init_huggingface(self, load_in_4bit: bool = False, load_in_8bit: bool = False):
        """
        Initialize via the shared HuggingFaceCausalLMLayer (fallback when
        vLLM isn't available) instead of loading transformers directly.

        Reusing it here - rather than the raw AutoModelForCausalLM/
        AutoTokenizer loading this used to do - gets this "production"
        entry point the same model-family-aware chat templating, Qwen3
        <think>-block stripping, and 4-bit/8-bit quantization support that
        experiments/ already relies on via common/llm_integration.py,
        instead of a second, independently-drifting reimplementation of
        all three.
        """
        self.hf_layer = HuggingFaceCausalLMLayer(
            LLMConfig(
                model_name=self.model_name,
                device="cuda",
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
            )
        )
        logger.info("Hugging Face model initialized via HuggingFaceCausalLMLayer")

    def _causal_task_instructions(self) -> str:
        """
        Instructions that get the model to emit the ANSWER:/CAUSAL_ASSERTIONS:
        structure _parse_response() below expects. Factored out so both
        backends can share it: the vLLM path folds it into a system message
        (see _build_causal_prompt), while the HF path folds it into the
        user-turn text instead and leaves system-message/chat-template/
        constraint-injection entirely to HuggingFaceCausalLMLayer.
        """
        return (
            'Answer in exactly ONE short sentence, in the form "X causes Y" '
            '(or "X does not cause Y"), naming only the two things directly '
            "involved. Use short noun phrases (2-4 words) for X and Y - no "
            'subordinate clauses ("which...", "because...", "that..."), no '
            "explanation of the mechanism, no intermediate steps - even if you "
            "know more about how or why. Only elaborate if the question "
            'explicitly asks "how" or "why".\n\n'
            "Then restate that same single relationship as one causal "
            "assertion, reusing the exact same noun phrases as your answer - "
            "not a paraphrase, not a restatement in different words, not an "
            "additional or more detailed claim.\n\n"
            "Format your response as:\n"
            "ANSWER: [X causes Y]\n"
            "CAUSAL_ASSERTIONS:\n"
            "- [X causes Y]"
        )

    def _build_causal_prompt(
        self, user_prompt: str, constraints: list[str] | None = None
    ) -> str:
        """
        Construct a fully chat-templated prompt for the vLLM path only.
        The HF path never calls this - HuggingFaceCausalLMLayer.generate()
        builds its own system message, applies its own model-family-aware
        chat template, and injects constraints itself.
        """
        system_prompt = (
            "You are a reasoning agent that generates responses grounded in "
            f"factual knowledge.\n\n{self._causal_task_instructions()}"
        )

        if constraints:
            constraint_text = "\n".join(f"- {c}" for c in constraints)
            system_prompt += (
                f"\n\nIMPORTANT: Avoid these contradictions:\n{constraint_text}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Format according to Llama-3 chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return formatted_prompt

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate response with causal assertions.

        Returns:
            Dict containing:
            - text: Generated response
            - causal_assertions: List of extracted assertions
            - metadata: Generation metadata
        """
        if config is None:
            config = GenerationConfig()

        if self.use_vllm:
            formatted_prompt = self._build_causal_prompt(prompt, constraints)
            result = await self._generate_vllm(formatted_prompt, config)
        else:
            # HuggingFaceCausalLMLayer builds its own system message, chat
            # template, and constraint injection - only the task text
            # (raw prompt + the ANSWER:/CAUSAL_ASSERTIONS: instructions)
            # needs to be assembled here.
            task_text = f"{prompt}\n\n{self._causal_task_instructions()}"
            result = await self._generate_huggingface(task_text, config, constraints)

        # Parse response to extract answer and causal assertions
        parsed = self._parse_response(result["text"])

        return {
            "text": parsed["answer"],
            "causal_assertions_raw": parsed["assertions"],
            "full_response": result["text"],
            "metadata": result.get("metadata", {}),
        }

    async def _generate_vllm(
        self, prompt: str, config: GenerationConfig
    ) -> dict[str, Any]:
        """Generate using vLLM engine"""
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
        )

        # Run in thread pool to avoid blocking
        outputs = await asyncio.to_thread(self.llm.generate, [prompt], sampling_params)

        output = outputs[0]
        generated_text = output.outputs[0].text

        return {
            "text": generated_text,
            "metadata": {
                "finish_reason": output.outputs[0].finish_reason,
                "tokens_generated": len(output.outputs[0].token_ids),
            },
        }

    async def _generate_huggingface(
        self,
        task_text: str,
        config: GenerationConfig,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate via the shared HuggingFaceCausalLMLayer (fallback path)"""
        generated_text = await asyncio.to_thread(
            self.hf_layer.generate,
            task_text,
            constraints,
            config.max_tokens,
            config.temperature,
            config.top_p,
        )

        return {"text": generated_text, "metadata": {}}

    def _parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse the structured response to extract answer and causal assertions.
        """
        lines = response.strip().split("\n")

        answer = ""
        assertions = []
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("ANSWER:"):
                current_section = "answer"
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("CAUSAL_ASSERTIONS:"):
                current_section = "assertions"
            elif line.startswith("-") and current_section == "assertions":
                assertion = line.lstrip("- ").strip()
                if assertion:
                    assertions.append(assertion)
            elif current_section == "answer" and not line.startswith("CAUSAL"):
                answer += " " + line

        return {"answer": answer.strip(), "assertions": assertions}

    def is_healthy(self) -> bool:
        """Check if engine is operational"""
        try:
            if self.use_vllm:
                return self.llm is not None
            else:
                return self.hf_layer is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

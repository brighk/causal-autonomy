"""
Module A: Inference Engine (Neural)
Hardware: NVIDIA A100/H100 GPU
Model: Llama-3-70B
Framework: PyTorch + vLLM for high-throughput inference
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import asyncio
from dataclasses import dataclass

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
        use_vllm: bool = True
    ):
        self.model_name = model_name
        self.use_vllm = use_vllm and VLLM_AVAILABLE

        logger.info(f"Initializing Inference Engine with {model_name}")

        if self.use_vllm:
            self._init_vllm(tensor_parallel_size, gpu_memory_utilization)
        else:
            logger.info("Using Hugging Face transformers (vLLM not available or disabled)")
            self._init_huggingface()

        logger.info("Inference Engine initialized successfully")

    def _init_vllm(self, tensor_parallel_size: int, gpu_memory_utilization: float):
        """Initialize vLLM engine for high-performance inference"""
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=4096
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM engine initialized")

    def _init_huggingface(self):
        """Initialize Hugging Face transformers (fallback)"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Hugging Face model initialized")

    def _build_causal_prompt(self, user_prompt: str, constraints: Optional[List[str]] = None) -> str:
        """
        Construct a prompt that encourages the model to generate
        verifiable causal assertions alongside the response.
        """
        system_prompt = """You are a reasoning agent that generates responses grounded in factual knowledge.

For each response:
1. Provide a clear, accurate answer
2. State the causal relationships or facts that support your answer
3. Use precise, verifiable statements

Format your response as:
ANSWER: [your response]
CAUSAL_ASSERTIONS:
- [assertion 1]
- [assertion 2]
..."""

        if constraints:
            constraint_text = "\n".join(f"- {c}" for c in constraints)
            system_prompt += f"\n\nIMPORTANT: Avoid these contradictions:\n{constraint_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Format according to Llama-3 chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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

        formatted_prompt = self._build_causal_prompt(prompt, constraints)

        if self.use_vllm:
            result = await self._generate_vllm(formatted_prompt, config)
        else:
            result = await self._generate_huggingface(formatted_prompt, config)

        # Parse response to extract answer and causal assertions
        parsed = self._parse_response(result['text'])

        return {
            'text': parsed['answer'],
            'causal_assertions_raw': parsed['assertions'],
            'full_response': result['text'],
            'metadata': result.get('metadata', {})
        }

    async def _generate_vllm(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Generate using vLLM engine"""
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty
        )

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.llm.generate([prompt], sampling_params)
        )

        output = outputs[0]
        generated_text = output.outputs[0].text

        return {
            'text': generated_text,
            'metadata': {
                'finish_reason': output.outputs[0].finish_reason,
                'tokens_generated': len(output.outputs[0].token_ids)
            }
        }

    async def _generate_huggingface(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Generate using Hugging Face transformers"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True
            )
        )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return {
            'text': generated_text,
            'metadata': {
                'tokens_generated': outputs.shape[1] - inputs['input_ids'].shape[1]
            }
        }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the structured response to extract answer and causal assertions.
        """
        lines = response.strip().split('\n')

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

        return {
            'answer': answer.strip(),
            'assertions': assertions
        }

    def is_healthy(self) -> bool:
        """Check if engine is operational"""
        try:
            if self.use_vllm:
                return self.llm is not None
            else:
                return self.model is not None and self.tokenizer is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

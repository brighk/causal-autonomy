"""
Baseline Methods for CAF Comparison
====================================
Implementations of baseline approaches to compare against CAF:
- Vanilla LLM (no enhancements)
- Chain of Thought (CoT) prompting
- Retrieval-Augmented Generation (RAG)

These baselines help demonstrate CAF's superiority in the paper.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import random

from experiments.caf_algorithm import InferenceLayer
from experiments.synthetic_dataset import CausalChain


@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""
    use_cot: bool = False  # Enable Chain of Thought
    use_rag: bool = False  # Enable RAG
    rag_top_k: int = 3     # Number of facts to retrieve for RAG
    cot_steps: int = 3     # Number of reasoning steps for CoT


class VanillaLLMBaseline(InferenceLayer):
    """
    Vanilla LLM baseline - single generation without enhancements.

    This is the simplest baseline: just ask the LLM directly.
    """

    def __init__(self, base_llm: InferenceLayer):
        self.base_llm = base_llm

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """Generate response without any enhancements."""
        # Ignore constraints for vanilla baseline
        return self.base_llm.generate(prompt, constraints=None)


class ChainOfThoughtBaseline(InferenceLayer):
    """
    Chain of Thought (CoT) baseline.

    Uses explicit step-by-step reasoning prompts to guide the LLM.
    This is a strong baseline used in many papers (Wei et al., 2022).

    Example:
        "Let's solve this step by step:
         1. First, identify...
         2. Then, analyze...
         3. Finally, conclude..."
    """

    def __init__(self, base_llm: InferenceLayer, num_steps: int = 3):
        self.base_llm = base_llm
        self.num_steps = num_steps

    def _create_cot_prompt(self, original_prompt: str) -> str:
        """Wrap prompt with Chain of Thought instructions."""
        cot_instruction = f"""Let's approach this step-by-step:

{original_prompt}

Please think through this carefully by breaking it down into {self.num_steps} clear reasoning steps:
1. First, identify the key elements and relationships
2. Then, analyze each causal link and verify logical consistency
3. Finally, provide your conclusion based on the reasoning

Show your reasoning for each step."""

        return cot_instruction

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """Generate with Chain of Thought prompting."""
        cot_prompt = self._create_cot_prompt(prompt)

        # Add constraints if provided (though vanilla CoT doesn't typically use them)
        if constraints:
            cot_prompt += f"\n\nAdditional considerations:\n"
            for i, constraint in enumerate(constraints, 1):
                cot_prompt += f"{i}. {constraint}\n"

        return self.base_llm.generate(cot_prompt, constraints=None)


class RAGBaseline(InferenceLayer):
    """
    Retrieval-Augmented Generation (RAG) baseline.

    Retrieves relevant facts from a knowledge base, then generates
    a response conditioned on those facts.

    This simulates standard RAG pipelines (Lewis et al., 2020) where:
    1. Query is used to retrieve top-k relevant documents
    2. Retrieved docs are added to prompt context
    3. LLM generates based on prompt + retrieved context
    """

    def __init__(
        self,
        base_llm: InferenceLayer,
        knowledge_base: Optional[Dict[str, List[str]]] = None,
        top_k: int = 3
    ):
        self.base_llm = base_llm
        self.knowledge_base = knowledge_base or {}
        self.top_k = top_k

    def set_knowledge_base(self, causal_chain: CausalChain):
        """
        Build a simple knowledge base from a causal chain.

        In a real RAG system, this would be:
        - Vector embeddings of documents
        - Semantic search with FAISS/ChromaDB
        - BM25 or dense retrieval

        For simulation, we use the ground truth facts from the chain.
        """
        domain = causal_chain.domain

        # Extract facts from the causal chain
        facts = []

        # Add each logical step as a fact
        for step in causal_chain.steps:
            # Convert step to natural language
            fact = step.to_natural_language()
            facts.append(fact)

        # Add ground truth entailments
        for entailment in causal_chain.ground_truth_entailments:
            facts.append(entailment)

        # Add contradiction info if present
        if causal_chain.injected_contradictions:
            for contradiction in causal_chain.injected_contradictions:
                facts.append(f"Note: {contradiction}")

        self.knowledge_base[domain] = facts

    def _retrieve_facts(self, prompt: str, domain: str = None) -> List[str]:
        """
        Retrieve top-k relevant facts from knowledge base.

        In a real system: semantic similarity search.
        For simulation: random sampling from the domain's facts.
        """
        if domain and domain in self.knowledge_base:
            available_facts = self.knowledge_base[domain]
        else:
            # Flatten all facts if domain unknown
            available_facts = []
            for facts in self.knowledge_base.values():
                available_facts.extend(facts)

        if not available_facts:
            return []

        # Retrieve top-k facts (randomly for simulation)
        k = min(self.top_k, len(available_facts))
        retrieved = random.sample(available_facts, k)

        return retrieved

    def _create_rag_prompt(
        self,
        original_prompt: str,
        retrieved_facts: List[str]
    ) -> str:
        """Create RAG prompt with retrieved context."""
        if not retrieved_facts:
            return original_prompt

        rag_prompt = "Based on the following relevant facts:\n\n"
        for i, fact in enumerate(retrieved_facts, 1):
            rag_prompt += f"{i}. {fact}\n"

        rag_prompt += f"\nNow, answer the following:\n{original_prompt}\n"
        rag_prompt += "\nProvide a response grounded in the facts above."

        return rag_prompt

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None,
        domain: str = None
    ) -> str:
        """Generate with retrieval-augmented prompting."""
        # Retrieve relevant facts
        retrieved_facts = self._retrieve_facts(prompt, domain)

        # Create RAG prompt
        rag_prompt = self._create_rag_prompt(prompt, retrieved_facts)

        # Add constraints if provided
        if constraints:
            rag_prompt += f"\n\nAdditional constraints:\n"
            for constraint in constraints:
                rag_prompt += f"- {constraint}\n"

        return self.base_llm.generate(rag_prompt, constraints=None)


class HybridRAGCoTBaseline(InferenceLayer):
    """
    Hybrid RAG + Chain of Thought baseline.

    Combines both approaches:
    1. Retrieve relevant facts (RAG)
    2. Reason step-by-step over retrieved facts (CoT)

    This is a very strong baseline representing state-of-the-art
    prompting techniques.
    """

    def __init__(
        self,
        base_llm: InferenceLayer,
        knowledge_base: Optional[Dict[str, List[str]]] = None,
        top_k: int = 3,
        num_steps: int = 3
    ):
        self.rag = RAGBaseline(base_llm, knowledge_base, top_k)
        self.cot_steps = num_steps

    def set_knowledge_base(self, causal_chain: CausalChain):
        """Build knowledge base from chain."""
        self.rag.set_knowledge_base(causal_chain)

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None,
        domain: str = None
    ) -> str:
        """Generate with both RAG and CoT."""
        # Retrieve facts
        retrieved_facts = self.rag._retrieve_facts(prompt, domain)

        # Create hybrid prompt
        hybrid_prompt = "Based on the following relevant facts:\n\n"
        for i, fact in enumerate(retrieved_facts, 1):
            hybrid_prompt += f"{i}. {fact}\n"

        hybrid_prompt += f"\n{prompt}\n\n"
        hybrid_prompt += f"Let's reason through this step-by-step using the facts above:\n"
        hybrid_prompt += "1. First, identify which facts are most relevant\n"
        hybrid_prompt += "2. Then, analyze how they connect logically\n"
        hybrid_prompt += "3. Finally, draw a conclusion grounded in the facts\n"

        if constraints:
            hybrid_prompt += f"\nConstraints to consider:\n"
            for constraint in constraints:
                hybrid_prompt += f"- {constraint}\n"

        return self.rag.base_llm.generate(hybrid_prompt, constraints=None)


# Factory functions for easy baseline creation

def create_vanilla_baseline(base_llm: InferenceLayer) -> InferenceLayer:
    """Create vanilla LLM baseline (no enhancements)."""
    return VanillaLLMBaseline(base_llm)


def create_cot_baseline(
    base_llm: InferenceLayer,
    num_steps: int = 3
) -> InferenceLayer:
    """Create Chain of Thought baseline."""
    return ChainOfThoughtBaseline(base_llm, num_steps)


def create_rag_baseline(
    base_llm: InferenceLayer,
    top_k: int = 3
) -> InferenceLayer:
    """Create RAG baseline."""
    return RAGBaseline(base_llm, top_k=top_k)


def create_rag_cot_baseline(
    base_llm: InferenceLayer,
    top_k: int = 3,
    num_steps: int = 3
) -> InferenceLayer:
    """Create hybrid RAG+CoT baseline (strongest baseline)."""
    return HybridRAGCoTBaseline(base_llm, top_k=top_k, num_steps=num_steps)


if __name__ == "__main__":
    """Demo of baseline methods."""
    print("CAF Baselines Demo")
    print("=" * 60)

    # In a real scenario, you'd use actual LLM here
    from experiments.caf_algorithm import SimulatedInferenceLayer
    base_llm = SimulatedInferenceLayer()

    prompt = "Explain how increased CO2 leads to global warming."

    print("\n1. Vanilla Baseline:")
    vanilla = create_vanilla_baseline(base_llm)
    print(vanilla.generate(prompt))

    print("\n2. Chain of Thought Baseline:")
    cot = create_cot_baseline(base_llm)
    print(cot.generate(prompt))

    print("\n3. RAG Baseline:")
    rag = create_rag_baseline(base_llm)
    # Normally would populate KB from data
    rag.knowledge_base = {
        "climate": [
            "CO2 is a greenhouse gas",
            "Greenhouse gases trap heat in atmosphere",
            "Increased heat leads to warming"
        ]
    }
    print(rag.generate(prompt, domain="climate"))

    print("\n4. RAG+CoT Baseline (Strongest):")
    hybrid = create_rag_cot_baseline(base_llm)
    hybrid.rag.knowledge_base = rag.knowledge_base
    print(hybrid.generate(prompt, domain="climate"))

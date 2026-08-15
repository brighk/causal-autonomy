"""
Synthetic Causal Chain Dataset Generator
=========================================
Generates synthetic logic chains with controlled complexity for
evaluating LLM reasoning capabilities and the CAF framework.

Features:
- Generates chains of varying depths (2-10 logical steps)
- Creates 2-3 prompt perturbations per chain (paraphrase, negation, noise)
- Includes ground truth for entailment verification
- Supports contradiction injection for stress testing
"""

import random
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from pathlib import Path
import re


class PerturbationType(Enum):
    """Types of prompt perturbations for semantic invariance testing."""
    PARAPHRASE = "paraphrase"
    DOUBLE_NEGATION = "double_negation"
    LEXICAL_NOISE = "lexical_noise"
    REORDERING = "reordering"
    SYNONYM_SUBSTITUTION = "synonym"


class RelationType(Enum):
    """Types of causal/logical relations in chains."""
    CAUSES = "causes"
    IMPLIES = "implies"
    ENABLES = "enables"
    PREVENTS = "prevents"
    REQUIRES = "requires"
    LEADS_TO = "leads_to"


@dataclass
class LogicalStep:
    """A single step in a causal chain."""
    antecedent: str
    consequent: str
    relation: RelationType
    confidence: float = 1.0
    step_number: int = 0

    def to_natural_language(self) -> str:
        """Convert to natural language statement."""
        relation_phrases = {
            RelationType.CAUSES: "causes",
            RelationType.IMPLIES: "implies that",
            RelationType.ENABLES: "enables",
            RelationType.PREVENTS: "prevents",
            RelationType.REQUIRES: "requires",
            RelationType.LEADS_TO: "leads to",
        }
        phrase = relation_phrases[self.relation]
        return f"{self.antecedent} {phrase} {self.consequent}"

    def to_rdf_triple(self) -> Tuple[str, str, str]:
        """Convert to RDF triple format."""
        subject = self.antecedent.lower().replace(" ", "_")
        predicate = f"caf:{self.relation.value}"
        obj = self.consequent.lower().replace(" ", "_")
        return (subject, predicate, obj)


@dataclass
class PromptPerturbation:
    """A perturbed version of a prompt for invariance testing."""
    original_prompt: str
    perturbed_prompt: str
    perturbation_type: PerturbationType
    expected_same_output: bool = True
    perturbation_details: Dict = field(default_factory=dict)


@dataclass
class CausalChain:
    """A complete causal chain with metadata."""
    chain_id: str
    steps: List[LogicalStep]
    domain: str
    complexity_score: float
    ground_truth_entailments: List[str]
    injected_contradictions: List[str] = field(default_factory=list)
    perturbations: List[PromptPerturbation] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        """Return the inference depth (number of logical steps)."""
        return len(self.steps)

    @property
    def initial_premise(self) -> str:
        """Return the initial premise of the chain."""
        return self.steps[0].antecedent if self.steps else ""

    @property
    def final_conclusion(self) -> str:
        """Return the final conclusion of the chain."""
        return self.steps[-1].consequent if self.steps else ""

    def to_prompt(self) -> str:
        """Convert chain to a reasoning prompt."""
        premises = [f"Given: {self.steps[0].antecedent}"]
        for step in self.steps:
            premises.append(f"- {step.to_natural_language()}")
        premises.append(f"\nQuestion: What can we conclude about {self.final_conclusion}?")
        return "\n".join(premises)

    def get_intermediate_conclusions(self) -> List[str]:
        """Get all intermediate conclusions in the chain."""
        return [step.consequent for step in self.steps]


class SyntheticDatasetGenerator:
    """
    Generates synthetic causal chain datasets for LLM evaluation.

    The generator creates logic chains across multiple domains with
    controlled complexity and perturbations for comprehensive testing.
    """

    # Domain-specific templates for generating realistic chains
    DOMAIN_TEMPLATES = {
        "physics": {
            "entities": [
                "temperature", "pressure", "volume", "energy", "mass",
                "velocity", "acceleration", "force", "momentum", "heat",
                "friction", "gravity", "resistance", "current", "voltage"
            ],
            "relations": [
                ("temperature increases", "pressure increases", RelationType.CAUSES),
                ("pressure increases", "volume decreases", RelationType.LEADS_TO),
                ("force applied", "acceleration occurs", RelationType.CAUSES),
                ("friction present", "heat generated", RelationType.LEADS_TO),
                ("mass increases", "gravitational pull strengthens", RelationType.CAUSES),
                ("voltage increases", "current increases", RelationType.LEADS_TO),
                ("resistance increases", "current decreases", RelationType.CAUSES),
                ("energy added", "temperature rises", RelationType.LEADS_TO),
                ("momentum conserved", "total momentum unchanged", RelationType.IMPLIES),
                ("heat transferred", "thermal equilibrium reached", RelationType.LEADS_TO),
            ]
        },
        "biology": {
            "entities": [
                "cell division", "protein synthesis", "DNA replication", "metabolism",
                "photosynthesis", "respiration", "enzyme activity", "mutation",
                "immune response", "hormone release", "neurotransmission", "gene expression"
            ],
            "relations": [
                ("DNA replication occurs", "cell division begins", RelationType.ENABLES),
                ("mutation happens", "protein structure changes", RelationType.CAUSES),
                ("enzyme activity increases", "reaction rate accelerates", RelationType.LEADS_TO),
                ("immune response triggered", "antibodies produced", RelationType.CAUSES),
                ("hormone released", "target cells activated", RelationType.LEADS_TO),
                ("photosynthesis occurs", "glucose produced", RelationType.CAUSES),
                ("respiration increases", "ATP generated", RelationType.LEADS_TO),
                ("gene expressed", "protein synthesized", RelationType.CAUSES),
                ("neurotransmitter released", "signal transmitted", RelationType.ENABLES),
                ("metabolism increases", "energy consumption rises", RelationType.LEADS_TO),
            ]
        },
        "economics": {
            "entities": [
                "demand", "supply", "price", "inflation", "unemployment",
                "interest rates", "GDP", "investment", "consumption", "exports",
                "imports", "currency value", "tax rates", "government spending"
            ],
            "relations": [
                ("demand increases", "price rises", RelationType.CAUSES),
                ("supply decreases", "price rises", RelationType.CAUSES),
                ("interest rates rise", "borrowing decreases", RelationType.LEADS_TO),
                ("inflation increases", "purchasing power decreases", RelationType.CAUSES),
                ("unemployment rises", "consumer spending falls", RelationType.LEADS_TO),
                ("GDP grows", "investment increases", RelationType.ENABLES),
                ("tax rates increase", "disposable income decreases", RelationType.CAUSES),
                ("exports increase", "currency strengthens", RelationType.LEADS_TO),
                ("government spending rises", "aggregate demand increases", RelationType.CAUSES),
                ("investment increases", "economic growth accelerates", RelationType.LEADS_TO),
            ]
        },
        "logic": {
            "entities": [
                "proposition P", "proposition Q", "proposition R", "proposition S",
                "statement A", "statement B", "condition X", "condition Y",
                "hypothesis H", "conclusion C", "premise M", "inference I"
            ],
            "relations": [
                ("P is true", "Q is true", RelationType.IMPLIES),
                ("Q is true", "R follows", RelationType.IMPLIES),
                ("A and B hold", "C is valid", RelationType.IMPLIES),
                ("X is satisfied", "Y is enabled", RelationType.ENABLES),
                ("H is assumed", "C can be derived", RelationType.ENABLES),
                ("M is established", "I is justified", RelationType.IMPLIES),
                ("not-P holds", "not-Q follows", RelationType.IMPLIES),
                ("P or Q holds", "at least one true", RelationType.IMPLIES),
                ("P and Q hold", "P holds", RelationType.IMPLIES),
                ("if P then Q, P holds", "Q holds", RelationType.IMPLIES),
            ]
        },
        "causality": {
            "entities": [
                "rain", "wet ground", "slippery roads", "traffic accidents",
                "flood", "crop damage", "food shortage", "price increase",
                "fire", "smoke", "alarm trigger", "evacuation",
                "storm", "power outage", "data loss", "business disruption"
            ],
            "relations": [
                ("rain falls", "ground becomes wet", RelationType.CAUSES),
                ("ground is wet", "roads become slippery", RelationType.LEADS_TO),
                ("roads are slippery", "accident risk increases", RelationType.CAUSES),
                ("heavy rain continues", "flooding occurs", RelationType.LEADS_TO),
                ("flooding occurs", "crops are damaged", RelationType.CAUSES),
                ("crops damaged", "food shortage develops", RelationType.LEADS_TO),
                ("fire starts", "smoke is produced", RelationType.CAUSES),
                ("smoke detected", "alarm triggers", RelationType.LEADS_TO),
                ("alarm triggers", "evacuation begins", RelationType.CAUSES),
                ("storm arrives", "power outage occurs", RelationType.CAUSES),
            ]
        }
    }

    # Paraphrase templates for perturbation
    PARAPHRASE_TEMPLATES = [
        ("causes", ["results in", "brings about", "produces", "triggers"]),
        ("implies that", ["suggests that", "means that", "indicates that"]),
        ("leads to", ["results in", "culminates in", "ends with"]),
        ("enables", ["allows", "permits", "makes possible"]),
        ("prevents", ["stops", "blocks", "inhibits"]),
        ("requires", ["needs", "demands", "necessitates"]),
    ]

    # Synonym mappings for lexical perturbation
    SYNONYMS = {
        "increases": ["rises", "grows", "elevates", "heightens"],
        "decreases": ["falls", "drops", "declines", "reduces"],
        "occurs": ["happens", "takes place", "transpires"],
        "begins": ["starts", "commences", "initiates"],
        "true": ["valid", "correct", "accurate"],
        "false": ["invalid", "incorrect", "inaccurate"],
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        self.chain_counter = 0

    def generate_chain(
        self,
        depth: int,
        domain: str,
        inject_contradiction: bool = False
    ) -> CausalChain:
        """
        Generate a single causal chain with specified depth and domain.

        Args:
            depth: Number of logical steps in the chain
            domain: Domain for the chain (physics, biology, economics, logic, causality)
            inject_contradiction: Whether to inject a contradiction for testing

        Returns:
            CausalChain object with steps and metadata
        """
        if domain not in self.DOMAIN_TEMPLATES:
            domain = random.choice(list(self.DOMAIN_TEMPLATES.keys()))

        templates = self.DOMAIN_TEMPLATES[domain]
        relations = templates["relations"]

        # Build chain by connecting relations
        steps = []
        used_relations = set()
        available_relations = list(relations)
        random.shuffle(available_relations)

        # Start with a random relation
        current_rel = available_relations.pop(0)
        steps.append(LogicalStep(
            antecedent=current_rel[0],
            consequent=current_rel[1],
            relation=current_rel[2],
            step_number=0
        ))
        used_relations.add(current_rel)

        # Build chain up to desired depth
        for i in range(1, depth):
            # Find a relation that can connect
            current_consequent = steps[-1].consequent

            # Try to find a connecting relation or generate synthetic connection
            found_connection = False
            for rel in available_relations:
                if rel not in used_relations:
                    # Use this relation with synthetic connection
                    steps.append(LogicalStep(
                        antecedent=current_consequent,
                        consequent=rel[1],
                        relation=rel[2],
                        step_number=i
                    ))
                    used_relations.add(rel)
                    found_connection = True
                    break

            if not found_connection:
                # Generate synthetic step
                entities = templates["entities"]
                new_consequent = random.choice(entities)
                relation_type = random.choice(list(RelationType))
                steps.append(LogicalStep(
                    antecedent=current_consequent,
                    consequent=f"{new_consequent} changes",
                    relation=relation_type,
                    step_number=i
                ))

        # Generate chain ID
        self.chain_counter += 1
        chain_id = f"chain_{domain}_{self.chain_counter:04d}"

        # Calculate complexity score based on depth and relation diversity
        unique_relations = len(set(step.relation for step in steps))
        complexity_score = (depth * 0.3) + (unique_relations * 0.2) + random.uniform(0, 0.5)

        # Generate ground truth entailments
        ground_truth = self._generate_ground_truth(steps)

        # Handle contradiction injection
        injected_contradictions = []
        if inject_contradiction:
            contradiction = self._inject_contradiction(steps)
            injected_contradictions.append(contradiction)

        return CausalChain(
            chain_id=chain_id,
            steps=steps,
            domain=domain,
            complexity_score=min(complexity_score, 1.0),
            ground_truth_entailments=ground_truth,
            injected_contradictions=injected_contradictions,
            metadata={
                "seed": self.seed,
                "generated_depth": depth,
                "actual_depth": len(steps),
            }
        )

    def _generate_ground_truth(self, steps: List[LogicalStep]) -> List[str]:
        """Generate ground truth entailments for a chain."""
        entailments = []

        # Direct entailments from each step
        for step in steps:
            entailments.append(f"If {step.antecedent}, then {step.consequent}")

        # Transitive entailments
        if len(steps) >= 2:
            entailments.append(
                f"If {steps[0].antecedent}, then ultimately {steps[-1].consequent}"
            )

        # Intermediate entailments for longer chains
        for i in range(len(steps) - 1):
            for j in range(i + 2, len(steps)):
                if random.random() < 0.3:  # Sample some intermediate entailments
                    entailments.append(
                        f"If {steps[i].antecedent}, then {steps[j].consequent}"
                    )

        return entailments

    def _inject_contradiction(self, steps: List[LogicalStep]) -> str:
        """Inject a contradiction into the chain."""
        if not steps:
            return ""

        # Choose a random step to contradict
        step = random.choice(steps)

        # Generate contradictory statement
        contradiction_templates = [
            f"{step.antecedent} does not cause {step.consequent}",
            f"{step.consequent} is independent of {step.antecedent}",
            f"The opposite of {step.consequent} occurs when {step.antecedent}",
        ]

        return random.choice(contradiction_templates)

    def generate_perturbations(
        self,
        chain: CausalChain,
        num_perturbations: int = 3
    ) -> List[PromptPerturbation]:
        """
        Generate prompt perturbations for semantic invariance testing.

        Args:
            chain: The causal chain to perturb
            num_perturbations: Number of perturbations to generate (2-3)

        Returns:
            List of PromptPerturbation objects
        """
        original_prompt = chain.to_prompt()
        perturbations = []

        perturbation_types = [
            PerturbationType.PARAPHRASE,
            PerturbationType.DOUBLE_NEGATION,
            PerturbationType.SYNONYM_SUBSTITUTION,
            PerturbationType.REORDERING,
        ]

        # Ensure we don't exceed available perturbation types
        num_perturbations = min(num_perturbations, len(perturbation_types))
        selected_types = random.sample(perturbation_types, num_perturbations)

        for ptype in selected_types:
            perturbed = self._apply_perturbation(original_prompt, chain, ptype)
            perturbations.append(perturbed)

        chain.perturbations = perturbations
        return perturbations

    def _apply_perturbation(
        self,
        prompt: str,
        chain: CausalChain,
        ptype: PerturbationType
    ) -> PromptPerturbation:
        """Apply a specific perturbation type to a prompt."""

        if ptype == PerturbationType.PARAPHRASE:
            perturbed = self._paraphrase(prompt)
            details = {"method": "relation_paraphrase"}

        elif ptype == PerturbationType.DOUBLE_NEGATION:
            perturbed = self._double_negate(prompt, chain)
            details = {"method": "double_negation", "target": "conclusion"}

        elif ptype == PerturbationType.SYNONYM_SUBSTITUTION:
            perturbed = self._substitute_synonyms(prompt)
            details = {"method": "synonym_substitution"}

        elif ptype == PerturbationType.REORDERING:
            perturbed = self._reorder_premises(prompt, chain)
            details = {"method": "premise_reordering"}

        else:
            perturbed = prompt
            details = {"method": "none"}

        return PromptPerturbation(
            original_prompt=prompt,
            perturbed_prompt=perturbed,
            perturbation_type=ptype,
            expected_same_output=True,
            perturbation_details=details
        )

    def _paraphrase(self, prompt: str) -> str:
        """Apply paraphrase perturbation."""
        result = prompt
        for original, alternatives in self.PARAPHRASE_TEMPLATES:
            if original in result:
                replacement = random.choice(alternatives)
                result = result.replace(original, replacement, 1)
                break
        return result

    def _double_negate(self, prompt: str, chain: CausalChain) -> str:
        """Apply double negation (P -> not(not(P)))."""
        conclusion = chain.final_conclusion
        if conclusion in prompt:
            double_neg = f"it is not the case that {conclusion} does not occur"
            return prompt.replace(conclusion, double_neg)
        return prompt

    def _substitute_synonyms(self, prompt: str) -> str:
        """Substitute words with synonyms."""
        result = prompt
        for word, synonyms in self.SYNONYMS.items():
            if word in result:
                replacement = random.choice(synonyms)
                result = result.replace(word, replacement, 1)
        return result

    def _reorder_premises(self, prompt: str, chain: CausalChain) -> str:
        """Reorder premises while maintaining logical validity."""
        lines = prompt.split("\n")
        premise_lines = [l for l in lines if l.startswith("- ")]
        other_lines = [l for l in lines if not l.startswith("- ")]

        if len(premise_lines) > 1:
            # Shuffle middle premises, keep first and last
            if len(premise_lines) > 2:
                middle = premise_lines[1:-1]
                random.shuffle(middle)
                premise_lines = [premise_lines[0]] + middle + [premise_lines[-1]]

        # Reconstruct
        result_lines = []
        premise_idx = 0
        for line in lines:
            if line.startswith("- ") and premise_idx < len(premise_lines):
                result_lines.append(premise_lines[premise_idx])
                premise_idx += 1
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def generate_dataset(
        self,
        num_chains: int = 75,
        min_depth: int = 2,
        max_depth: int = 10,
        perturbations_per_chain: int = 3,
        contradiction_rate: float = 0.2,
        domains: Optional[List[str]] = None
    ) -> List[CausalChain]:
        """
        Generate a complete synthetic dataset.

        Args:
            num_chains: Number of causal chains to generate (50-100)
            min_depth: Minimum chain depth
            max_depth: Maximum chain depth
            perturbations_per_chain: Number of perturbations per chain (2-3)
            contradiction_rate: Fraction of chains with injected contradictions
            domains: List of domains to use (None for all)

        Returns:
            List of CausalChain objects with perturbations
        """
        if domains is None:
            domains = list(self.DOMAIN_TEMPLATES.keys())

        dataset = []

        for i in range(num_chains):
            # Vary depth across chains
            depth = random.randint(min_depth, max_depth)
            domain = domains[i % len(domains)]
            inject_contradiction = random.random() < contradiction_rate

            # Generate chain
            chain = self.generate_chain(
                depth=depth,
                domain=domain,
                inject_contradiction=inject_contradiction
            )

            # Generate perturbations
            self.generate_perturbations(chain, perturbations_per_chain)

            dataset.append(chain)

        return dataset

    def export_dataset(
        self,
        dataset: List[CausalChain],
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export dataset to file.

        Args:
            dataset: List of CausalChain objects
            output_path: Path to output file
            format: Output format (json, jsonl, csv)

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "metadata": {
                    "num_chains": len(dataset),
                    "total_perturbations": sum(len(c.perturbations) for c in dataset),
                    "domains": list(set(c.domain for c in dataset)),
                    "depth_range": (
                        min(c.depth for c in dataset),
                        max(c.depth for c in dataset)
                    ),
                    "seed": self.seed,
                },
                "chains": [self._chain_to_dict(c) for c in dataset]
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "jsonl":
            with open(output_path, "w") as f:
                for chain in dataset:
                    f.write(json.dumps(self._chain_to_dict(chain)) + "\n")

        return str(output_path)

    def _chain_to_dict(self, chain: CausalChain) -> Dict:
        """Convert a CausalChain to a dictionary."""
        return {
            "chain_id": chain.chain_id,
            "domain": chain.domain,
            "depth": chain.depth,
            "complexity_score": chain.complexity_score,
            "initial_premise": chain.initial_premise,
            "final_conclusion": chain.final_conclusion,
            "prompt": chain.to_prompt(),
            "steps": [
                {
                    "step_number": s.step_number,
                    "antecedent": s.antecedent,
                    "consequent": s.consequent,
                    "relation": s.relation.value,
                    "natural_language": s.to_natural_language(),
                }
                for s in chain.steps
            ],
            "ground_truth_entailments": chain.ground_truth_entailments,
            "injected_contradictions": chain.injected_contradictions,
            "perturbations": [
                {
                    "type": p.perturbation_type.value,
                    "original": p.original_prompt,
                    "perturbed": p.perturbed_prompt,
                    "expected_same_output": p.expected_same_output,
                    "details": p.perturbation_details,
                }
                for p in chain.perturbations
            ],
            "metadata": chain.metadata,
        }


def generate_paper_dataset(output_dir: str = "experiments/data") -> str:
    """
    Generate the canonical dataset for the paper.

    Creates 75 logic chains with 3 perturbations each:
    - 50-100 range satisfied (75 chains)
    - 2-3 perturbations each (3 perturbations)
    - Controlled contradiction injection (20%)
    - Multiple domains for diversity

    Returns:
        Path to the generated dataset file
    """
    generator = SyntheticDatasetGenerator(seed=42)

    dataset = generator.generate_dataset(
        num_chains=75,
        min_depth=2,
        max_depth=10,
        perturbations_per_chain=3,
        contradiction_rate=0.2,
        domains=["physics", "biology", "economics", "logic", "causality"]
    )

    output_path = f"{output_dir}/synthetic_causal_chains.json"
    generator.export_dataset(dataset, output_path, format="json")

    print(f"Generated dataset with {len(dataset)} chains")
    print(f"Total perturbations: {sum(len(c.perturbations) for c in dataset)}")
    print(f"Chains with contradictions: {sum(1 for c in dataset if c.injected_contradictions)}")
    print(f"Depth range: {min(c.depth for c in dataset)} - {max(c.depth for c in dataset)}")
    print(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_paper_dataset()

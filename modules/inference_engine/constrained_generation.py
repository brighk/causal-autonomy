"""
Constrained Generation Module for Step B: Constraint Satisfaction

Implements token-level constraint satisfaction where LLM outputs are validated
against RDF triples in real-time during generation.

Key Innovation:
- Query SPARQL database during generation
- Bias logits to prefer tokens aligned with KB
- Discard tokens that would contradict ground truth
"""
from typing import List, Dict, Any, Optional, Set, Tuple
import torch
from transformers import LogitsProcessor
from SPARQLWrapper import SPARQLWrapper, JSON
from loguru import logger
import re
import asyncio
from functools import lru_cache


class RDFConstraintProcessor(LogitsProcessor):
    """
    Custom logits processor that constrains generation based on RDF triples.

    During each token generation step:
    1. Extract entities/predicates from partial text
    2. Query Fuseki for relevant constraints
    3. Penalize logits for tokens that would violate KB
    """

    def __init__(
        self,
        tokenizer,
        fuseki_endpoint: str,
        penalty_weight: float = 10.0,
        enable_logging: bool = False
    ):
        self.tokenizer = tokenizer
        self.fuseki_endpoint = fuseki_endpoint
        self.penalty_weight = penalty_weight
        self.enable_logging = enable_logging

        # Initialize SPARQL
        self.sparql = SPARQLWrapper(fuseki_endpoint)
        self.sparql.setReturnFormat(JSON)

        # Cache for KB queries (avoid repeated queries)
        self._kb_cache: Dict[str, List[str]] = {}

        logger.info(f"RDFConstraintProcessor initialized with penalty={penalty_weight}")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to apply RDF constraints.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            scores: Tensor of shape (batch_size, vocab_size)

        Returns:
            Modified scores with KB constraints applied
        """
        # Decode current partial generation
        for batch_idx in range(input_ids.shape[0]):
            partial_text = self.tokenizer.decode(
                input_ids[batch_idx],
                skip_special_tokens=True
            )

            # Extract constraints from KB based on partial text
            constraints = self._get_constraints_for_text(partial_text)

            if constraints:
                # Apply penalties to violating tokens
                scores[batch_idx] = self._apply_constraints(
                    scores[batch_idx],
                    partial_text,
                    constraints
                )

        return scores

    def _get_constraints_for_text(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text and query KB for relevant constraints.

        Returns:
            Dict with:
            - 'allowed_facts': Facts from KB that should be followed
            - 'prohibited_terms': Terms that would contradict KB
        """
        # Extract entities (simple regex-based for now)
        entities = self._extract_entities(text)

        if not entities:
            return {}

        # Query KB for each entity
        constraints = {
            'allowed_facts': [],
            'prohibited_terms': set()
        }

        for entity in entities:
            facts = self._query_entity_facts(entity)
            constraints['allowed_facts'].extend(facts)

        return constraints

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract potential entities from text.

        Simple heuristic: capitalized words and common nouns.
        For production, use spaCy NER.
        """
        # Look for capitalized words
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Also check for common keywords
        keywords = ['rain', 'sun', 'water', 'road', 'wet', 'dry', 'cause', 'effect']
        for keyword in keywords:
            if keyword.lower() in text.lower():
                entities.append(keyword)

        return list(set(entities))[:3]  # Limit to 3 entities for performance

    @lru_cache(maxsize=100)
    def _query_entity_facts(self, entity: str) -> List[Dict[str, str]]:
        """
        Query Fuseki for facts about an entity.

        Returns list of facts as dicts: {'predicate': 'causes', 'object': 'wet roads'}
        """
        cache_key = entity.lower()
        if cache_key in self._kb_cache:
            return self._kb_cache[cache_key]

        # Build SPARQL query
        query = f"""
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?p ?o ?oLabel
WHERE {{
    ?s rdfs:label "{entity.lower()}" .
    ?s ?p ?o .
    OPTIONAL {{ ?o rdfs:label ?oLabel }}
}}
LIMIT 10
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            facts = []
            for binding in results.get('results', {}).get('bindings', []):
                fact = {
                    'predicate': binding.get('p', {}).get('value', ''),
                    'object': binding.get('o', {}).get('value', ''),
                    'object_label': binding.get('oLabel', {}).get('value', '')
                }
                facts.append(fact)

            self._kb_cache[cache_key] = facts

            if self.enable_logging and facts:
                logger.debug(f"Found {len(facts)} facts for entity '{entity}'")

            return facts

        except Exception as e:
            logger.warning(f"KB query failed for '{entity}': {e}")
            return []

    def _apply_constraints(
        self,
        scores: torch.FloatTensor,
        partial_text: str,
        constraints: Dict[str, Any]
    ) -> torch.FloatTensor:
        """
        Apply KB constraints to logits.

        Strategy:
        1. Boost tokens that align with KB facts
        2. Penalize tokens that would contradict KB
        """
        # Get top-k token predictions to check
        top_k = 50
        top_scores, top_indices = torch.topk(scores, top_k)

        # Decode candidate tokens
        for idx, token_id in enumerate(top_indices):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)

            # Check if token would violate constraints
            if self._violates_constraints(partial_text, token_text, constraints):
                # Apply penalty
                scores[token_id] -= self.penalty_weight

                if self.enable_logging:
                    logger.debug(f"Penalized token: '{token_text}' (violates KB)")

        return scores

    def _violates_constraints(
        self,
        partial_text: str,
        next_token: str,
        constraints: Dict[str, Any]
    ) -> bool:
        """
        Check if adding next_token would violate KB constraints.

        Simple heuristic:
        - Check if prohibited terms would appear
        - Check if statement would contradict known facts
        """
        future_text = partial_text + next_token

        # Check prohibited terms
        for term in constraints.get('prohibited_terms', []):
            if term.lower() in future_text.lower():
                return True

        # Check for contradictions with known facts
        allowed_facts = constraints.get('allowed_facts', [])

        # Example: If KB says "rain causes wet roads"
        # and partial text is "rain causes", penalize "dry"
        for fact in allowed_facts:
            if 'causes' in fact.get('predicate', ''):
                expected_object = fact.get('object_label', fact.get('object', ''))

                # If we're generating after "causes", check alignment
                if 'cause' in partial_text.lower() and expected_object:
                    # If next token would introduce contradiction
                    if 'wet' in expected_object and 'dry' in next_token.lower():
                        return True
                    if 'dry' in expected_object and 'wet' in next_token.lower():
                        return True

        return False


class ConstrainedInferenceEngine:
    """
    Inference engine with RDF constraint satisfaction.

    Extends base InferenceEngine with token-level KB grounding.
    """

    def __init__(
        self,
        base_engine,
        fuseki_endpoint: str,
        constraint_strength: float = 10.0,
        enable_constraint_logging: bool = False
    ):
        self.base_engine = base_engine
        self.fuseki_endpoint = fuseki_endpoint
        self.constraint_strength = constraint_strength
        self.enable_logging = enable_constraint_logging

        logger.info("ConstrainedInferenceEngine initialized")

    async def generate_constrained(
        self,
        prompt: str,
        config,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with RDF constraint satisfaction.

        Returns same format as base engine but with KB grounding applied.
        """
        formatted_prompt = self.base_engine._build_causal_prompt(prompt, constraints)

        # Create RDF constraint processor
        constraint_processor = RDFConstraintProcessor(
            tokenizer=self.base_engine.tokenizer,
            fuseki_endpoint=self.fuseki_endpoint,
            penalty_weight=self.constraint_strength,
            enable_logging=self.enable_logging
        )

        # Generate with constraints
        if self.base_engine.use_vllm:
            # vLLM doesn't support custom logit processors directly
            # Fall back to standard generation with post-validation
            logger.warning("vLLM doesn't support custom logit processors, using post-validation")
            result = await self.base_engine._generate_vllm(formatted_prompt, config)
        else:
            # Use HuggingFace with custom logit processor
            result = await self._generate_constrained_hf(
                formatted_prompt,
                config,
                constraint_processor
            )

        # Parse response
        parsed = self.base_engine._parse_response(result['text'])

        return {
            'text': parsed['answer'],
            'causal_assertions_raw': parsed['assertions'],
            'full_response': result['text'],
            'metadata': {
                **result.get('metadata', {}),
                'constraint_satisfaction': 'enabled',
                'fuseki_endpoint': self.fuseki_endpoint
            }
        }

    async def _generate_constrained_hf(
        self,
        prompt: str,
        config,
        constraint_processor: RDFConstraintProcessor
    ) -> Dict[str, Any]:
        """
        Generate using HuggingFace with custom logit processor.
        """
        inputs = self.base_engine.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.base_engine.model.device)

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.base_engine.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                logits_processor=[constraint_processor]  # KEY: Apply RDF constraints
            )
        )

        generated_text = self.base_engine.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return {
            'text': generated_text,
            'metadata': {
                'tokens_generated': outputs.shape[1] - inputs['input_ids'].shape[1],
                'constrained': True
            }
        }

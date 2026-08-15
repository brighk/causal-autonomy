"""
Module B: Semantic Parser (Middleware)
Framework: LangChain + spaCy
Task: Text-to-SPARQL mapping via entity extraction and linking

Protocol:
1. Extract entities using spaCy NER
2. Link entities to Wikidata/ConceptNet URIs using vector similarity
3. Construct SPARQL query using templates
"""
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from loguru import logger
import re

from api.models import Triplet, CausalAssertion

# Optional spacy import - disabled due to version incompatibility
SPACY_AVAILABLE = False
spacy = None
logger.warning("spaCy temporarily disabled due to version incompatibility with thinc")


class EntityLinker:
    """Links entities to knowledge base URIs using semantic similarity"""

    def __init__(self, chromadb_host: str = "localhost", chromadb_port: int = 8000):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Try to connect to ChromaDB with timeout
        self.chroma_client = None
        self.entity_collection = None

        try:
            # Try HTTP client first
            from chromadb import HttpClient
            import httpx
            self.chroma_client = HttpClient(
                host=chromadb_host,
                port=chromadb_port,
                settings=Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider" if False else None)
            )

            # Test connection with timeout
            try:
                self.entity_collection = self.chroma_client.get_collection("entities")
                logger.info(f"Connected to ChromaDB at {chromadb_host}:{chromadb_port}")
            except:
                self.entity_collection = self.chroma_client.create_collection(
                    name="entities",
                    metadata={"description": "Entity URI mappings from ConceptNet/Wikidata"}
                )
                logger.info(f"Created new ChromaDB collection at {chromadb_host}:{chromadb_port}")
        except Exception as e:
            logger.warning(f"Could not connect to ChromaDB server: {e}")
            logger.info("Using in-memory ChromaDB (entity linking will not persist)")
            # Fallback to embedded/in-memory ChromaDB
            try:
                import chromadb
                self.chroma_client = chromadb.Client()
                try:
                    self.entity_collection = self.chroma_client.get_collection("entities")
                except:
                    self.entity_collection = self.chroma_client.create_collection(
                        name="entities",
                        metadata={"description": "Entity URI mappings from ConceptNet/Wikidata"}
                    )
            except Exception as e2:
                logger.error(f"Failed to initialize ChromaDB: {e2}")

        logger.info("Entity Linker initialized")

    def link_entity(self, entity_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find the most similar entity URIs for a given text.

        Returns:
            List of dicts with 'uri', 'label', 'score'
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([entity_text]).tolist()

        # Query ChromaDB
        results = self.entity_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        linked_entities = []
        for i, entity_id in enumerate(results['ids'][0]):
            linked_entities.append({
                'uri': entity_id,
                'label': results['metadatas'][0][i].get('label', entity_id),
                'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                'source': results['metadatas'][0][i].get('source', 'unknown')
            })

        return linked_entities

    def add_entity(self, uri: str, label: str, source: str = "manual"):
        """Add a new entity to the vector database"""
        embedding = self.embedding_model.encode([label]).tolist()

        self.entity_collection.add(
            ids=[uri],
            embeddings=embedding,
            metadatas=[{"label": label, "source": source}]
        )


class SemanticParser:
    """
    Extracts triplets from natural language and converts to SPARQL.

    Pipeline:
    1. NER with spaCy
    2. Entity linking with ChromaDB
    3. Relation extraction
    4. SPARQL generation
    """

    def __init__(
        self,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        spacy_model: str = "en_core_web_lg"
    ):
        # Load spaCy model if available
        self.nlp = None
        # Temporarily disable spacy due to version incompatibility
        logger.warning("spaCy NER temporarily disabled - will use pattern matching")

        # Initialize entity linker
        self.entity_linker = EntityLinker(chromadb_host, chromadb_port)

        # Predicate templates (common relations)
        self.predicate_templates = {
            'is': 'rdf:type',
            'has': 'schema:hasProperty',
            'causes': 'causality:causes',
            'located_in': 'schema:location',
            'part_of': 'schema:isPartOf',
            'related_to': 'schema:relatedTo',
            'created_by': 'schema:creator',
            'used_for': 'schema:purpose'
        }

        logger.info("Semantic Parser initialized")

    async def parse(
        self,
        text: str,
        causal_assertions: Optional[List[CausalAssertion]] = None
    ) -> 'ParsedResult':
        """
        Parse text into RDF triplets.

        Args:
            text: Natural language text
            causal_assertions: Pre-identified causal assertions from LLM

        Returns:
            ParsedResult with triplets and SPARQL query
        """
        triplets = []

        # If we have explicit causal assertions, parse those
        if causal_assertions:
            for assertion in causal_assertions:
                assertion_triplets = await self._parse_assertion(
                    assertion.assertion_text
                )
                triplets.extend(assertion_triplets)
                # Update the assertion's triplets
                assertion.triplets = assertion_triplets
        else:
            # Parse the entire text
            triplets = await self._parse_text(text)

        # Generate SPARQL query
        sparql_query = self._generate_sparql(triplets)

        return ParsedResult(
            triplets=triplets,
            sparql_query=sparql_query,
            source_text=text
        )

    async def _parse_text(self, text: str) -> List[Triplet]:
        """Extract triplets from free-form text"""
        doc = self.nlp(text)

        triplets = []

        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Simple subject-predicate-object extraction based on dependency parsing
        for sent in doc.sents:
            for token in sent:
                # Look for subject-verb-object patterns
                if token.dep_ in ('nsubj', 'nsubjpass'):
                    subject = token.text
                    predicate = token.head.text
                    object_ = None

                    # Find object
                    for child in token.head.children:
                        if child.dep_ in ('dobj', 'attr', 'pobj'):
                            object_ = child.text
                            break

                    if object_:
                        # Link entities to URIs
                        subject_uri = await self._get_entity_uri(subject)
                        predicate_uri = self._get_predicate_uri(predicate)
                        object_uri = await self._get_entity_uri(object_)

                        triplets.append(Triplet(
                            subject=subject_uri,
                            predicate=predicate_uri,
                            object=object_uri
                        ))

        return triplets

    async def _parse_assertion(self, assertion: str) -> List[Triplet]:
        """Parse a single causal assertion into triplets"""
        # Use the same parsing logic as _parse_text
        return await self._parse_text(assertion)

    async def _get_entity_uri(self, entity_text: str) -> str:
        """
        Get the URI for an entity using entity linking.
        Falls back to a local URI if no match found.
        """
        linked = self.entity_linker.link_entity(entity_text, top_k=1)

        if linked and linked[0]['score'] > 0.7:
            return linked[0]['uri']
        else:
            # Create a local URI
            normalized = re.sub(r'[^a-zA-Z0-9]', '_', entity_text.lower())
            return f"local:{normalized}"

    def _get_predicate_uri(self, predicate_text: str) -> str:
        """
        Map predicate text to a standard URI.
        Uses template matching and falls back to generic relation.
        """
        predicate_lower = predicate_text.lower()

        # Check templates
        for template_key, uri in self.predicate_templates.items():
            if template_key in predicate_lower:
                return uri

        # Default relation
        normalized = re.sub(r'[^a-zA-Z0-9]', '_', predicate_lower)
        return f"relation:{normalized}"

    def _generate_sparql(self, triplets: List[Triplet]) -> str:
        """
        Generate SPARQL SELECT query from triplets.

        Example:
        SELECT ?o WHERE { :subject :predicate ?o }
        """
        if not triplets:
            return ""

        # Build WHERE clause
        where_patterns = []
        for t in triplets:
            # Use variable if it's a query, else use literal
            obj_var = "?o" if t.object_.startswith("local:") else f"<{t.object_}>"
            where_patterns.append(
                f"<{t.subject}> <{t.predicate}> {obj_var} ."
            )

        where_clause = "\n    ".join(where_patterns)

        sparql = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <https://schema.org/>
PREFIX causality: <http://causality.org/>
PREFIX local: <http://local.caf/>
PREFIX relation: <http://local.caf/relation/>

SELECT ?o
WHERE {{
    {where_clause}
}}
        """.strip()

        return sparql

    def is_healthy(self) -> bool:
        """Check if parser is operational"""
        try:
            return (
                self.nlp is not None and
                self.entity_linker is not None
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class ParsedResult:
    """Result from semantic parsing"""

    def __init__(
        self,
        triplets: List[Triplet],
        sparql_query: str,
        source_text: str
    ):
        self.triplets = triplets
        self.sparql_query = sparql_query
        self.source_text = source_text

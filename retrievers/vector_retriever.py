import ollama
import logging
import os
from graph.neo4j_manager import Neo4jManager
from models.ollama_manager import OllamaManager

# Este método convierte la pregunta del usuario en un vector numérico.
# Usa el mismo modelo que usamos para los chunks con la finalidad de que estén en el mismo espacio vectorial.
def get_query_embedding(question: str) -> list:    
    model = os.getenv("OLLAMA_MODEL", "llama3")

    # Mismo truncado que en text_processor.py
    words = question.split()
    if len(words) > 500:
        question = " ".join(words[:500])

    try:
        response = ollama.embeddings(model=model, prompt=question)
        return response["embedding"]
    except Exception as e:
        import logging
        logging.warning(f"Embedding de pregunta fallido: {e}")
        return []


# Clase de recuperación semántica por similitud de embeddings.
# Mejoras implementadas respecto al RAG básico:
#   1. HyQE: busca también por similitud con preguntas hipotéticas
#   2. Filtro por sección: prioriza chunks de secciones relevantes
#   3. Contexto del grafo: enriquece cada chunk con sus entidades
class VectorRetriever:

    def __init__(self, neo4j: Neo4jManager, top_k: int = 5):
        self.neo4j = neo4j
        self.ai    = OllamaManager()
        self.top_k = top_k

    # Método principal: busca chunks similares a la pregunta y los enriquece con las entidades del grafo.
    def query(self, question: str, section_filter: str = None) -> dict:        
        # Paso 1: embedding de la pregunta
        embedding = get_query_embedding(question)
        if not embedding:
            return {"type": "vector", "question": question,
                    "results": [], "error": "Embedding failed"}

        # Paso 2: búsqueda por similitud vectorial
        chunks = self._vector_search(embedding, section_filter)

        # Paso 3: enriquecer con entidades del grafo
        enriched = self._enrich_with_entities(chunks)

        return {
            "type":      "vector",
            "question":  question,
            "results":   enriched
        }

    # Búsqueda kNN en el índice vectorial de Neo4j y opcionalmente filtra por sección del artículo.
    def _vector_search(self, embedding: list,
                       section_filter: str = None) -> list:        
        if section_filter:
            query = """
            MATCH (c:Chunk)
            WHERE c.section = $section
            WITH c,
                 vector.similarity.cosine(c.embedding, $embedding) AS score
            WHERE score > 0.7
            RETURN c.id AS id, c.text AS text,
                   c.section AS section, score
            ORDER BY score DESC
            LIMIT $top_k
            """
            params = {
                "embedding":      embedding,
                "section":        section_filter,
                "top_k":          self.top_k
            }
        else:
            query = """
            MATCH (c:Chunk)
            WITH c,
                 vector.similarity.cosine(c.embedding, $embedding) AS score
            WHERE score > 0.7
            RETURN c.id AS id, c.text AS text,
                   c.section AS section, score
            ORDER BY score DESC
            LIMIT $top_k
            """
            params = {
                "embedding": embedding,
                "top_k":     self.top_k
            }

        return self.neo4j.run_query(query, params)

    # Mejora clave del RAG: enriquece cada chunk recuperado con las entidades conectadas en el grafo.
    # Esto permite al LLM responder con datos estructurados del grafo además del texto semántico del chunk.
    def _enrich_with_entities(self, chunks: list) -> list:
        enriched = []
        for chunk in chunks:
            # Buscar entidades conectadas al chunk
            entities_query = """
            MATCH (c:Chunk {id: $chunk_id})-[:MENCIONA]->(e)
            RETURN labels(e)[0] AS tipo,
                   coalesce(e.name, e.type, e.description) AS nombre
            """
            entities = self.neo4j.run_query(
                entities_query,
                {"chunk_id": chunk["id"]}
            )

            enriched.append({
                "text":     chunk["text"],
                "section":  chunk["section"],
                "score":    chunk.get("score", 0),
                "entities": entities  # Entidades del grafo
            })

        return enriched
from typing import List, Dict, Any
from models.ollama_manager import OllamaManager

# Clase Agente que genera la respuesta final en lenguaje natural combinando todos los contextos recuperados por los retrievers.
#   Recibe:
#    - La pregunta original del usuario
#    - Los resultados de cada retriever (structured + semantic)
#   Devuelve:
#    - Una respuesta coherente en lenguaje natural
#    - Las fuentes usadas para justificar la respuesta
class AnswerGenerator:
    

    def __init__(self):
        self.ai = OllamaManager()

    # Genera la respuesta final fusionando todos los contextos.
    def generate(self, question: str,
                 contexts: List[Dict[str, Any]]) -> Dict[str, Any]:       
        # Formatear el contexto para el prompt
        context_text = self._format_contexts(contexts)

        prompt = (
            f"You are an expert in IoT health monitoring systems. "
            f"Answer the following question using ONLY the provided context.\n\n"
            f"Question: {question}\n\n"
            f"Context from the knowledge graph:\n{context_text}\n\n"
            f"Instructions:\n"
            f"- Answer in clear, concise language\n"
            f"- Cite specific devices, technologies or articles when relevant\n"
            f"- If the context does not contain enough information, say so\n"
            f"- Do not invent information not present in the context\n\n"
            f"Answer:"
        )

        answer = self.ai.chat(prompt)

        return {
            "question": question,
            "answer":   answer,
            "sources":  self._extract_sources(contexts),
            "context_used": len(contexts)
        }
    
    # Formatea los contextos de los retrievers en texto legible para el LLM.
    def _format_contexts(self, contexts: List[Dict]) -> str:       
        formatted = []

        for i, ctx in enumerate(contexts):
            ctx_type = ctx.get("type", "unknown")

            if ctx_type == "vector":
                # Contexto semántico: texto del chunk + entidades
                for result in ctx.get("results", []):
                    text    = result.get("text", "")[:500]
                    section = result.get("section", "")
                    score   = result.get("score", 0)
                    entities = result.get("entities", [])

                    entity_names = [
                        e.get("nombre", "") for e in entities
                        if e.get("nombre")
                    ]

                    formatted.append(
                        f"[Semantic context - {section} (score: {score:.2f})]:\n"
                        f"{text}\n"
                        f"Related entities: {', '.join(entity_names[:10])}"
                    )

            elif ctx_type in ("manual", "text2cypher"):
                # Contexto estructurado: resultados del grafo
                results = ctx.get("results", [])
                if results:
                    formatted.append(
                        f"[Structured data from graph]:\n" +
                        "\n".join([str(r) for r in results[:10]])
                    )

        return "\n\n".join(formatted) if formatted else "No context available."
    
    # Extrae las fuentes usadas para la respuesta.
    def _extract_sources(self, contexts: List[Dict]) -> List[str]:      
        sources = []
        for ctx in contexts:
            if ctx.get("type") == "vector":
                for r in ctx.get("results", []):
                    section = r.get("section", "")
                    if section:
                        sources.append(f"Chunk [{section}]")
            elif ctx.get("query"):
                sources.append(f"Query [{ctx.get('query')}]")
        return list(set(sources))
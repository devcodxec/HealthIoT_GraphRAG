from typing import List
from pydantic import BaseModel, Field
from models.ollama_manager import OllamaManager

# Una subpregunta atómica derivada de la pregunta original.
class SubQuery(BaseModel):    
    question: str = Field(
        description="A specific atomic question that can be answered independently"
    )
    retriever_hint: str = Field(
        description="Suggested retriever: 'manual', 'structured', or 'semantic'"
    )

# Resultado de la descomposición de una pregunta compleja.
class DecomposedQuery(BaseModel):    
    is_complex: bool = Field(
        description="True if the question needs decomposition, False if it is already atomic"
    )
    sub_queries: List[SubQuery] = Field(default_factory=list)

# Agente que descompone preguntas complejas en subpreguntas atómicas.
#   Por qué es necesario:
#   Una pregunta como "Qué sistemas usan CNN con Edge computing y cuáles son sus limitaciones?" tiene tres partes distintas:
#       1. ¿Qué sistemas usan CNN?         → structured
#       2. ¿Qué sistemas usan Edge?        → structured
#       3. ¿Cuáles son sus limitaciones?   → semantic
#   Cada parte necesita un retriever diferente. Sin descomposición ningún retriever puede responder bien la pregunta completa.
class QueryDecomposer:

    def __init__(self):
        self.ai = OllamaManager()

    def decompose(self, question: str) -> DecomposedQuery:
        result = self.ai.structured_output(
            prompt=(
                f"Analyze this question about IoT health monitoring.\n\n"
                f"Question: {question}\n\n"
                f"Rules for decomposition:\n"
                f"- ONLY decompose if the question asks about 2 or more "
                f"DIFFERENT entities or concepts\n"
                f"- Example of complex: 'Which articles use CNN AND Edge computing "
                f"AND what are their limitations?' → 3 sub-questions\n"
                f"- Example of simple: 'What devices monitor ECG?' → NO decomposition\n"
                f"- Do NOT generate sub-questions about implicit assumptions\n"
                f"- Do NOT generate sub-questions about obvious properties\n"
                f"- Maximum 3 sub-questions\n"
                f"- Each sub-question must be directly answerable from a "
                f"scientific paper database"
            ),
            schema=DecomposedQuery,
            system_prompt=(
                "You are an expert at analyzing research questions about "
                "IoT health monitoring systems. Be conservative: only decompose "
                "when truly necessary. Most questions are already atomic."
            )
        )
        return result

    # Devuelve lista de subpreguntas para el router.
    # Si la pregunta es simple, devuelve la pregunta original.
    def get_sub_questions(self, question: str) -> List[dict]:       
        decomposed = self.decompose(question)

        if not decomposed.is_complex or not decomposed.sub_queries:
            # Pregunta simple → devolver tal cual
            return [{"question": question, "retriever_hint": "auto"}]

        # Pregunta compleja → devolver subpreguntas
        return [
            {
                "question": sq.question,
                "retriever_hint": sq.retriever_hint
            }
            for sq in decomposed.sub_queries
        ]
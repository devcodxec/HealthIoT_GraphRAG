from pydantic import BaseModel, Field
from typing import Optional
from models.ollama_manager import OllamaManager

#  Clase para representar la evaluación del agente crítico.
class CriticEvaluation(BaseModel):
    resolves_question: bool = Field(
        description="True if the answer resolves the user's original question"
    )
    is_faithful: bool = Field(
        description="True if the answer is faithful to the provided context"
    )
    score: float = Field(
        description="Overall quality score from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
    feedback: str = Field(
        description="Specific feedback on what is missing or incorrect"
    )
    needs_retry: bool = Field(
        description="True if the answer should be regenerated"
    )

# Clase Agente crítico que evalúa si la respuesta generada resuelve la intención original del usuario.
# Evalúa dos dimensiones:
#   1. Resolución: responde la pregunta que hizo el usuario?
#   2. Fidelidad: está basada en el contexto del grafo?
# Si la evaluación falla (score < 0.6), indica que hay que reintentar.
class CriticAgent:   

    RETRY_THRESHOLD = 0.4  # score mínimo para aceptar la respuesta

    def __init__(self):
        self.ai = OllamaManager()

    # Evalúa la calidad de la respuesta generada.
    def evaluate(self, question: str, answer: str,
                 context: str) -> CriticEvaluation:
       
        result = self.ai.structured_output(
            prompt=(
                f"Evaluate this answer for the given question.\n\n"
                f"Original question: {question}\n\n"
                f"Generated answer: {answer}\n\n"
                f"Context used: {context[:800]}\n\n"
                f"Evaluate:\n"
                f"1. resolves_question: Does the answer address what was asked?\n"
                f"2. is_faithful: Is the answer based only on the context?\n"
                f"3. score: Overall quality 0.0-1.0\n"
                f"4. feedback: What is missing or incorrect?\n"
                f"5. needs_retry: Should the answer be regenerated?"
            ),
            schema=CriticEvaluation,
            system_prompt=(
                "You are a strict evaluator of answers about IoT health monitoring. "
                "Be critical and precise. A good answer must directly address "
                "the question and be supported by the provided context."
            )
        )
        return result

    # Determina si la respuesta es aceptable.
    def is_acceptable(self, evaluation: CriticEvaluation) -> bool:
        return (
            evaluation.score >= self.RETRY_THRESHOLD and
            evaluation.resolves_question and
            not evaluation.needs_retry
        )
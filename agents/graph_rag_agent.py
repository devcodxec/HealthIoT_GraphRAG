from typing import Dict, Any
from graph.neo4j_manager import Neo4jManager
from retrievers.retriever_router import RetrieverRouter
from agents.query_decomposer import QueryDecomposer
from agents.answer_generator import AnswerGenerator
from agents.critic_agent import CriticAgent


# Clase Agente principal que orquesta todo el sistema.
# Flujo completo:
#   1. Recibe pregunta del usuario
#   2. QueryDecomposer la descompone si es compleja
#   3. RetrieverRouter busca en el grafo por cada subpregunta
#   4. AnswerGenerator fusiona contextos y genera respuesta
#   5. CriticAgent evalúa la respuesta
#   6. Si falla → reintenta (máximo 2 veces) 
class GraphRAGAgent:    

    MAX_RETRIES = 2

    def __init__(self, neo4j: Neo4jManager):
        self.router     = RetrieverRouter(neo4j)
        self.decomposer = QueryDecomposer()
        self.generator  = AnswerGenerator()
        self.critic     = CriticAgent()

    # Método principal: procesa la pregunta de principio a fin.
    def run(self, question: str) -> Dict[str, Any]:        
        print(f"\n{'='*60}")
        print(f"Pregunta: {question}")
        print(f"{'='*60}")

        # ── Paso 1: descomponer la pregunta ───────────────────────
        print("\n[1] Descomponiendo la pregunta...")
        sub_questions = self.decomposer.get_sub_questions(question)
        print(f"  → {len(sub_questions)} subpregunta(s) generada(s)")
        for sq in sub_questions:
            print(f"     • {sq['question']}")

        # ── Paso 2: recuperar contexto para cada subpregunta ──────
        print("\n[2] Recuperando contexto del grafo...")
        contexts = []
        for sq in sub_questions:
            result = self.router.route(sq["question"])
            contexts.append(result)
            print(f"  → [{result.get('type')}] "
                  f"{len(result.get('results', []))} resultados")
        
        if contexts and contexts[0].get("type") == "greeting":
            return {
                "question":      question,
                "answer":        contexts[0].get("answer", ""),
                "sources":       [],
                "evaluation": {
                    "score":             1.0,
                    "resolves_question": True,
                    "is_faithful":       True,
                    "feedback":          "Greeting handled directly."
                },
                "sub_questions": [question],
                "retries":       0,
            }

        # ── Paso 3: generar respuesta ─────────────────────────────
        print("\n[3] Generando respuesta...")
        answer_data = self.generator.generate(question, contexts)
        print(f"  → Respuesta generada ({len(answer_data['answer'])} chars)")

        # ── Paso 4: evaluar con el agente crítico ─────────────────
        print("\n[4] Evaluando respuesta con agente crítico...")

        context_preview = str(contexts)[:500]
        evaluation = self.critic.evaluate(
            question=question,
            answer=answer_data["answer"],
            context=context_preview
        )

        print(f"  → Score: {evaluation.score:.2f}")
        print(f"  → Resuelve la pregunta: {evaluation.resolves_question}")
        print(f"  → Fiel al contexto: {evaluation.is_faithful}")
        print(f"  → Feedback: {evaluation.feedback}")

        # ── Paso 5: reintentar si la evaluación falla ─────────────
        retry_count = 0
        while (not self.critic.is_acceptable(evaluation) and
               retry_count < self.MAX_RETRIES):

            retry_count += 1
            print(f"\n  ↻ Reintento {retry_count}/{self.MAX_RETRIES}...")
            print(f"  Motivo: {evaluation.feedback}")

            # Regenerar con el feedback del crítico en el prompt
            answer_data = self.generator.generate(
                question=f"{question}\n\nNote: {evaluation.feedback}",
                contexts=contexts
            )

            evaluation = self.critic.evaluate(
                question=question,
                answer=answer_data["answer"],
                context=context_preview
            )
            print(f"  → Nuevo score: {evaluation.score:.2f}")

        # ── Resultado final ───────────────────────────────────────
        return {
            "question":   question,
            "answer":     answer_data["answer"],
            "sources":    answer_data["sources"],
            "evaluation": {
                "score":              evaluation.score,
                "resolves_question":  evaluation.resolves_question,
                "is_faithful":        evaluation.is_faithful,
                "feedback":           evaluation.feedback,
            },
            "sub_questions": [sq["question"] for sq in sub_questions],
            "retries":       retry_count,
        }
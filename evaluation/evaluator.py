from typing import List, Dict, Any
from agents.graph_rag_agent import GraphRAGAgent
import time

# Preguntas de evaluación del enunciado del proyecto
EVALUATION_QUERIES = [

    # SEMÁNTICAS:
    # Buscan conceptos, ideas y contexto en el texto de los artículos
    {
        "question": "What are the most recurring ethical and technical "
                    "limitations in the use of wearable devices?",
        "type": "semantic",
        "expected_entities": ["Challenge", "Device"]
    },
    {
        "question": "What technical recommendations do authors provide "
                    "for ensuring interoperability of IoT data protocols?",
        "type": "semantic",
        "expected_entities": ["Recommendation", "Technology"]
    },
    {
        "question": "What are the main advantages of using Edge computing "
                    "in real-time health monitoring systems?",
        "type": "semantic",
        "expected_entities": ["Strength", "Architecture"]
    },
    {
        "question": "What future research directions are proposed "
                    "by authors in IoT health monitoring papers?",
        "type": "semantic",
        "expected_entities": ["FutureWork", "AIModel"]
    },
    {
        "question": "What privacy and security concerns are identified "
                    "in wearable health monitoring systems?",
        "type": "semantic",
        "expected_entities": ["Challenge", "Device", "Technology"]
    },

    # ESTRUCTURADAS:
    # Buscan datos exactos del grafo: listas, conteos, filtros
    {
        "question": "List all devices that capture ECG signals",
        "type": "structured",
        "expected_entities": ["Device", "Signal"]
    },
    {
        "question": "What wireless technologies are most used "
                    "in health monitoring systems?",
        "type": "structured",
        "expected_entities": ["Technology"]
    },
    {
        "question": "What AI models are used for signal classification "
                    "in wearable IoT systems?",
        "type": "structured",
        "expected_entities": ["AIModel", "Signal"]
    },
    {
        "question": "Which articles propose Edge or Fog architecture "
                    "for patient monitoring?",
        "type": "structured",
        "expected_entities": ["Architecture", "Article"]
    },
    {
        "question": "What are the most critical challenges identified "
                    "in IoT health monitoring literature?",
        "type": "structured",
        "expected_entities": ["Challenge"]
    },

    # HÍBRIDAS:
    # Combinan búsqueda semántica y estructurada
    {
        "question": "What articles use AI techniques with explainability "
                    "methods to analyze physiological signals?",
        "type": "hybrid",
        "expected_entities": ["AIModel", "Signal", "Article"]
    },
    {
        "question": "What systems use LSTM or CNN models combined "
                    "with Bluetooth or WiFi technology?",
        "type": "hybrid",
        "expected_entities": ["AIModel", "Technology", "Device"]
    },
    {
        "question": "What devices capture SpO2 signals and what "
                    "architecture do they use for data transmission?",
        "type": "hybrid",
        "expected_entities": ["Device", "Signal", "Architecture"]
    },
    {
        "question": "What are the strengths and limitations of systems "
                    "that use deep learning for ECG analysis?",
        "type": "hybrid",
        "expected_entities": ["Strength", "Challenge", "AIModel", "Signal"]
    },
    {
        "question": "What Big Data tools are used alongside machine learning "
                    "models in IoT health monitoring systems?",
        "type": "hybrid",
        "expected_entities": ["BigData", "AIModel", "Architecture"]
    },
]


#   Evaluación estructurada del sistema Graph RAG.
#   Métricas implementadas:
#       1. Critic Score    → puntuación del agente crítico (0-1)
#       2. Resolution Rate → % de preguntas resueltas correctamente
#       3. Faithfulness    → % de respuestas fieles al contexto
#       4. Coverage        → % de entidades esperadas encontradas
#       5. Retry Rate      → % de preguntas que necesitaron reintento
class Evaluator:


    PAUSE_BETWEEN_QUESTIONS = 30  # Pausa en segundos entre preguntas para evitar sobrecarga


    def __init__(self, agent: GraphRAGAgent):
        self.agent = agent

    # Ejecuta la evaluación completa sobre todas las preguntas.
    def run_evaluation(self) -> Dict[str, Any]:

        print("\n" + "="*60)
        print("EVALUACIÓN ESTRUCTURADA DEL SISTEMA")
        print("="*60)

        results = []

        for i, query in enumerate(EVALUATION_QUERIES):
            print(f"\n[{i+1}/{len(EVALUATION_QUERIES)}] {query['type'].upper()}")
            result = self.agent.run(query["question"])
            result["query_type"]          = query["type"]
            result["expected_entities"]   = query["expected_entities"]
            results.append(result)

        # Pausa de 30 segundos entre preguntas para enfriar la CPU
        if i < len(EVALUATION_QUERIES) - 1:
            print(f"  ⏸ Pausa de enfriamiento 30s...")
            time.sleep(self.PAUSE_BETWEEN_QUESTIONS)

        # Calcular métricas
        metrics = self._calculate_metrics(results)
        self._print_report(metrics, results)

        return {"results": results, "metrics": metrics}

    # Calcula las métricas de evaluación.
    def _calculate_metrics(self,
                           results: List[Dict]) -> Dict[str, float]:
        total = len(results)
        if total == 0:
            return {}

        scores           = [r["evaluation"]["score"] for r in results]
        resolved         = [r["evaluation"]["resolves_question"]
                            for r in results]
        faithful         = [r["evaluation"]["is_faithful"]
                            for r in results]
        retries          = [r["retries"] for r in results]

        return {
            "avg_critic_score":  sum(scores) / total,
            "resolution_rate":   sum(resolved) / total,
            "faithfulness_rate": sum(faithful) / total,
            "retry_rate":        sum(1 for r in retries if r > 0) / total,
            "avg_retries":       sum(retries) / total,
            "total_questions":   total,
        }

    # Imprime el reporte final de evaluación.
    def _print_report(self, metrics: Dict, results: List[Dict]):
        print("\n" + "="*60)
        print("REPORTE FINAL DE EVALUACIÓN")
        print("="*60)
        print(f"Total preguntas evaluadas:  {metrics['total_questions']}")
        print(f"Critic Score promedio:      {metrics['avg_critic_score']:.2%}")
        print(f"Tasa de resolución:         {metrics['resolution_rate']:.2%}")
        print(f"Tasa de fidelidad:          {metrics['faithfulness_rate']:.2%}")
        print(f"Tasa de reintento:          {metrics['retry_rate']:.2%}")

        # Desglose por tipo de pregunta
        print("\nDesglose por tipo de pregunta:")
        for qtype in ["semantic", "structured", "hybrid"]:
            subset = [r for r in results if r["query_type"] == qtype]
            if not subset:
                continue
            avg = sum(r["evaluation"]["score"] for r in subset) / len(subset)
            resolved = sum(
                1 for r in subset if r["evaluation"]["score"] >= 0.4
            )
            print(f"\n  [{qtype.upper()}] — {len(subset)} preguntas")
            print(f"    Score promedio: {avg:.2%}")
            print(f"    Resueltas:      {resolved}/{len(subset)}")

        # Detalle individual de cada pregunta
        print("\nDetalle por pregunta:")
        for r in results:
            score    = r["evaluation"]["score"]
            qtype    = r["query_type"]
            resolved = "✓" if score >= 0.4 else "✗"
            faithful = "✓" if r["evaluation"]["is_faithful"] else "✗"
            print(
                f"  [{qtype}] {resolved} score={score:.2f} "
                f"fiel={faithful} | {r['question'][:55]}..."
            )
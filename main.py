import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.neo4j_manager import Neo4jManager
from extraction.text_processor import TextProcessor
from source.pdf_extractor import load_pdfs_from_folder

def run_pipeline():

    # 1. Conexión Neo4j
    db = Neo4jManager()
    db.test_connection()
    db.setup_schema()

    # 2. Procesador
    processor = TextProcessor(neo4j_manager=db)

    # 3. Carpeta de PDFs de Scopus
    scopus_folder = os.path.join(
        os.path.dirname(__file__), "data", "scopus"
    )

    # 4. Cargar todos los PDFs
    papers = load_pdfs_from_folder(scopus_folder)
    print(f"\nTotal de artículos a procesar: {len(papers)}")

    # 5. Procesar cada artículo
    for i, paper in enumerate(papers):
        print(f"\n[{i+1}/{len(papers)}]")
        try:
            processor.process_document(
                text=paper["text"],
                metadata=paper["metadata"]
            )
        except Exception as e:
            print(f"  Error procesando {paper['metadata']['title']}: {e}")
            continue  # Si falla uno, continúa con el siguiente

    print("\nPipeline completado.")
    db.close()

# Prueba los tres retrievers con preguntas de ejemplo.
def test_retrievers():
    
    from retrievers.retriever_router import RetrieverRouter

    db = Neo4jManager()
    router = RetrieverRouter(neo4j=db)

    # Preguntas de prueba del enunciado
    questions = [
        # Estructurada → Text2Cypher o Manual
        "List all devices that capture ECG signals",
        # Semántica → Vectorial
        "What are the ethical limitations of wearable devices?",
        # Manual → ManualRetriever
        "What are the most used wireless technologies?",
    ]

    for question in questions:
        print(f"\n{'='*50}")
        print(f"Pregunta: {question}")
        result = router.route(question)
        print(f"Retriever usado: {result['type']}")
        print(f"Resultados: {len(result.get('results', []))} items")
        for r in result.get("results", [])[:3]:
            print(f"  → {r}")

    db.close()

# Prueba completa del agente Graph RAG con preguntas de ejemplo.
def test_agent():
    from agents.graph_rag_agent import GraphRAGAgent

    db    = Neo4jManager()
    agent = GraphRAGAgent(neo4j=db)

    # Probamos las siguientes preguntas, que cubren los 3 tipos:
    questions = [
        "What devices are used for ECG monitoring in IoT systems?",
        "What are the main challenges in wearable IoT health monitoring?",
        "What AI models are used for signal classification?",
    ]

    for question in questions:
        result = agent.run(question)
        print(f"\nRespuesta: {result['answer'][:300]}")
        print(f"Score:     {result['evaluation']['score']:.2f}")
        print(f"Resuelve:  {result['evaluation']['resolves_question']}")
        print(f"Fuentes:   {result['sources']}")

    db.close()

# Evaluación estructurada completa del sistema.
def run_evaluation():
    from agents.graph_rag_agent import GraphRAGAgent
    from evaluation.evaluator import Evaluator

    db        = Neo4jManager()
    agent     = GraphRAGAgent(neo4j=db)
    evaluator = Evaluator(agent)
    report    = evaluator.run_evaluation()
    db.close()
    return report


if __name__ == "__main__":
    # Uso: python main.py [comando]
    comando = sys.argv[1] if len(sys.argv) > 1 else "help"

    if comando == "pipeline":
        run_pipeline()

    elif comando == "retrievers":
        test_retrievers()

    elif comando == "agent":
        test_agent()

    elif comando == "eval":
        run_evaluation()

    else:
        print("\nUso: python main.py [comando]")
        print("\nComandos disponibles:")
        print("  pipeline    → construir el grafo desde PDFs")
        print("  retrievers  → probar los 3 retrievers")
        print("  agent       → probar el agente completo")
        print("  eval        → evaluación estructurada completa")
    
# Interfaz Streamlit para HealthIoT GraphRAG
import streamlit as st
import sys
import os
import time
from graph.neo4j_manager import Neo4jManager
from extraction.text_processor import TextProcessor
from source.pdf_extractor import load_pdfs_from_folder


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthIoT GraphRAG",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CONEXIÓN A NEO4J Y CONFIGURACIÓN DE SISTEMA
# ─────────────────────────────────────────────────────────────

# Crea la conexión a Neo4j una sola vez.
@st.cache_resource
def get_neo4j():    
    from dotenv import load_dotenv
    load_dotenv(override=True)
    return Neo4jManager()

# Crea el router una sola vez.
@st.cache_resource
def get_router(_db):    
    from retrievers.retriever_router import RetrieverRouter
    return RetrieverRouter(neo4j=_db)

# Crea el agente una sola vez.
@st.cache_resource
def get_agent(_db):
    
    from agents.graph_rag_agent import GraphRAGAgent
    return GraphRAGAgent(neo4j=_db)


# ─────────────────────────────────────────────────────────────
# SIDEBAR Y NAVEGACIÓN
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("HealthIoT GraphRAG")
    st.caption("Sistema de análisis bibliográfico IoT Salud")
    st.divider()

    page = st.radio(
        "Selecciona una opción:",
        options=[
            "⚙️  Generar Grafo",
            "🔍  Retrievers",
            "🦾  Agentes",
            "📊  Evaluación",
            "🗄️  Estado del Grafo",
        ]
    )

    st.divider()

    # Verificación de la conexión Neo4j
    try:
        db = get_neo4j()
        db.driver.verify_connectivity()
        st.success("Neo4j conectado")

        # Conteo rápido de nodos
        count = db.run_query("MATCH (n) RETURN count(n) AS total")
        total = count[0]["total"] if count else 0
        st.metric("Nodos en el grafo", total)

    except Exception:
        st.error("Neo4j desconectado")


# ─────────────────────────────────────────────────────────────
# PÁGINA 1 — Generar Grafo de Conocimiento
# ─────────────────────────────────────────────────────────────
if "Generar" in page:
    st.title("⚙️ Generación del Grafo de Conocimiento")
    st.caption("Procesa los PDFs de Scopus y construye el grafo en Neo4j.")

    # Mostrar PDFs disponibles
    scopus_folder = os.path.join(os.path.dirname(__file__), "data", "scopus")
    if os.path.exists(scopus_folder):
        pdfs = sorted([f for f in os.listdir(scopus_folder) if f.endswith(".pdf")])
        st.info(f"PDFs encontrados en data/scopus/: **{len(pdfs)}**")
        for pdf in pdfs:
            st.text(f"  📄 {pdf}")
    else:
        st.warning("Carpeta data/scopus/ no encontrada")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Iniciar pipeline", type="primary"):
            db = get_neo4j()
            processor = TextProcessor(neo4j_manager=db)
            papers    = load_pdfs_from_folder(scopus_folder)

            progress = st.progress(0)
            status   = st.empty()

            for i, paper in enumerate(papers):
                status.text(f"Procesando: {paper['metadata']['title'][:50]}...")
                try:
                    processor.process_document(
                        text=paper["text"],
                        metadata=paper["metadata"]
                    )
                except Exception as e:
                    st.error(f"Error en {paper['metadata']['title']}: {e}")
                progress.progress((i + 1) / len(papers))

                # Pausa de 2 minutos entre artículos para enfriar la CPU
                if i < len(papers) - 1:
                    status.text(f"Pausa de enfriamiento... ({i+1}/{len(papers)} completados)")
                    time.sleep(120)

            status.text("✅ Pipeline completado")
            st.success(f"Se procesaron {len(papers)} artículos correctamente.")
            st.rerun()

    with col2:
        if st.button("🗑️ Limpiar grafo", type="secondary"):
            db = get_neo4j()
            db.run_query("MATCH (n) DETACH DELETE n")
            st.warning("Grafo limpiado.")
            st.rerun()


# ─────────────────────────────────────────────────────────────
# PÁGINA 2 — Ejecutar Retrievers
# ─────────────────────────────────────────────────────────────
elif "Retrievers" in page:
    st.title("🔍 Prueba de Retrievers")
    st.caption("Comparativa de estrategias: Text2Cypher, Vectorial y Manual.")

    db     = get_neo4j()
    router = get_router(db)

    # Selector de retriever
    retriever_type = st.selectbox(
        "Tipo de retriever:",
        ["Automático (Router)", "Text2Cypher", "Vector RAG", "Manual"]
    )

    # Input de pregunta
    question = st.text_input(
        "Escribe tu pregunta:",
        placeholder="What devices are used for ECG monitoring?"
    )

    # Preguntas de ejemplo
    st.caption("Ejemplos de preguntas:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📡 Dispositivos ECG"):
            question = "List all devices that capture ECG signals"
    with col2:
        if st.button("📶 Tecnologías"):
            question = "What are the most used wireless technologies?"
    with col3:
        if st.button("⚠️ Desafíos"):
            question = "What are the main challenges in IoT health monitoring?"

    if question and st.button("🔍 Buscar", type="primary"):
        with st.spinner("Buscando en el grafo..."):
            result = router.route(question)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retriever usado", result.get("type", "N/A"))
        with col2:
            st.metric("Resultados", len(result.get("results", [])))

        st.subheader("Resultados")

        if result.get("type") == "welcome":
            st.success(result.get("answer", ""))
        else:
            results = result.get("results", [])
            if results:
                for i, r in enumerate(results[:10]):
                    with st.expander(f"Resultado {i+1}"):
                        if "text" in r:
                            st.text(r["text"][:300])
                            if r.get("entities"):
                                st.caption("Entidades relacionadas:")
                                for e in r["entities"][:5]:
                                    st.text(
                                        f"  {e.get('tipo')}: {e.get('nombre')}"
                                    )
                        else:
                            st.json(r)
            else:
                st.warning("No se encontraron resultados.")


# ─────────────────────────────────────────────────────────────
# PÁGINA 3 — Agente GraphRAG
# ─────────────────────────────────────────────────────────────
elif "Agente" in page:
    st.title("🦾 Agente GraphRAG")
    st.caption("Integración de Router, Descomposición de Consultas y Agente Crítico.")

    db    = get_neo4j()
    agent = get_agent(db)

    # Historial de conversación
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mostrar historial
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{entry['score']:.2f}")
            col2.metric("Resuelve", "✓" if entry["resolves"] else "✗")
            col3.metric("Reintentos", entry["retries"])

    # Input de pregunta
    question = st.chat_input("Escribe tu pregunta sobre el corpus IoT salud...")

    # Preguntas de ejemplo en sidebar
    st.sidebar.divider()
    st.sidebar.caption("Preguntas de ejemplo:")
    example_questions = [
        "What devices monitor ECG in IoT systems?",
        "What are the main challenges in wearable health monitoring?",
        "What AI models are used for signal classification?",
        "What wireless technologies are most used?",
        "What future work trends are proposed in the papers?",
    ]
    for eq in example_questions:
        if st.sidebar.button(eq[:40] + "...", key=eq):
            question = eq

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("El agente está procesando tu pregunta..."):
                result = agent.run(question)

            # Respuesta principal
            st.write(result["answer"])

            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Score crítico",
                        f"{result['evaluation']['score']:.2f}")
            col2.metric("Resuelve",
                        "✓" if result["evaluation"]["resolves_question"]
                        else "✗")
            col3.metric("Fiel al contexto",
                        "✓" if result["evaluation"]["is_faithful"] else "✗")
            col4.metric("Reintentos", result["retries"])

            # Detalle expandible
            with st.expander("Ver detalle completo"):
                st.subheader("Subpreguntas generadas")
                for sq in result.get("sub_questions", []):
                    st.text(f"• {sq}")

                st.subheader("Fuentes utilizadas")
                for src in result.get("sources", []):
                    st.text(f"• {src}")

                st.subheader("Feedback del agente crítico")
                st.info(result["evaluation"]["feedback"])

        # Guardar en historial
        st.session_state.chat_history.append({
            "question": question,
            "answer":   result["answer"],
            "score":    result["evaluation"]["score"],
            "resolves": result["evaluation"]["resolves_question"],
            "retries":  result["retries"],
        })

# ─────────────────────────────────────────────────────────────
# PÁGINA 4 — Evaluación estructurada
# ─────────────────────────────────────────────────────────────
elif "Evaluación" in page:
    st.title("📊 Evaluación Estructurada")
    st.caption("Métricas de calidad del sistema Graph RAG.")

    if st.button("▶️ Ejecutar evaluación completa", type="primary"):
        db    = get_neo4j()
        agent = get_agent(db)

        from evaluation.evaluator import Evaluator, EVALUATION_QUERIES

        evaluator = Evaluator(agent)
        progress  = st.progress(0)
        status    = st.empty()    
        results   = []

        for i, query in enumerate(EVALUATION_QUERIES):
            st.text(f"Evaluando: {query['question'][:60]}...")
            result = agent.run(query["question"])
            result["query_type"] = query["type"]
            results.append(result)
            progress.progress((i + 1) / len(EVALUATION_QUERIES))

            if i < len(EVALUATION_QUERIES) - 1:
                for remaining in range(30, 0, -5):
                    status.text(
                        f"⏸ Enfriando CPU... "
                        f"siguiente pregunta en {remaining}s "
                        f"({i+1}/{len(EVALUATION_QUERIES)} completadas)"
                    )
                    time.sleep(5)

        # Métricas globales
        scores    = [r["evaluation"]["score"] for r in results]
        resolved  = [r["evaluation"]["score"] >= 0.4 for r in results]
        faithful  = [r["evaluation"]["is_faithful"] for r in results]

        st.divider()
        st.subheader("Métricas globales")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score promedio",
                    f"{sum(scores)/len(scores):.2%}")
        col2.metric("Tasa resolución",
                    f"{sum(resolved)/len(resolved):.2%}")
        col3.metric("Tasa fidelidad",
                    f"{sum(faithful)/len(faithful):.2%}")
        col4.metric("Preguntas evaluadas", len(results))

        # Desglose por tipo de pregunta
        st.divider()
        st.subheader("Desglose por tipo de pregunta")
        for qtype in ["semantic", "structured", "hybrid"]:
            subset = [r for r in results if r["query_type"] == qtype]
            if not subset:
                continue
            avg       = sum(r["evaluation"]["score"] for r in subset) / len(subset)
            resueltas = sum(1 for r in subset if r["evaluation"]["score"] >= 0.4)
            icon      = "🔵" if qtype == "semantic" else "🟢" if qtype == "structured" else "🟡"
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{icon} [{qtype.upper()}] Score", f"{avg:.2%}")
            col2.metric("Resueltas", f"{resueltas}/{len(subset)}")
            col3.metric("Preguntas", len(subset))

        # Detalle individual de cada pregunta
        st.subheader("Detalle por pregunta")
        for r in results:
            score   = r["evaluation"]["score"]
            color   = "✅" if score >= 0.6 else "⚠️" if score >= 0.4 else "❌"
            with st.expander(
                f"{color} [{r['query_type']}] {r['question'][:60]}..."
            ):
                st.write(f"**Respuesta:** {r['answer'][:300]}")
                col1, col2 = st.columns(2)
                col1.metric("Score", f"{score:.2f}")
                col2.metric("Reintentos", r["retries"])
                st.info(f"Feedback: {r['evaluation']['feedback']}")


# ─────────────────────────────────────────────────────────────
# PÁGINA 5 — Estado del grafo
# ─────────────────────────────────────────────────────────────
elif "Estado" in page:
    st.title("🗄️ Estado del Grafo Neo4j")

    db = get_neo4j()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nodos por tipo")
        results = db.run_query("""
            MATCH (n)
            RETURN labels(n)[0] AS tipo, count(n) AS cantidad
            ORDER BY cantidad DESC
        """)
        if results:
            for r in results:
                st.metric(r["tipo"], r["cantidad"])
        else:
            st.warning("El grafo está vacío")

    with col2:
        st.subheader("Relaciones por tipo")
        results = db.run_query("""
            MATCH ()-[r]->()
            RETURN type(r) AS tipo, count(r) AS cantidad
            ORDER BY cantidad DESC
        """)
        if results:
            for r in results:
                st.metric(r["tipo"], r["cantidad"])
        else:
            st.warning("No hay relaciones")

    st.divider()
    st.subheader("Muestra de entidades")

    entity_type = st.selectbox(
        "Ver entidades de tipo:",
        ["Device", "Signal", "Technology", "AIModel",
         "Architecture", "Challenge", "Recommendation"]
    )

    sample = db.run_query(f"""
        MATCH (e:{entity_type})
        RETURN coalesce(e.name, e.type, e.description) AS nombre,
               e.description AS descripcion
        LIMIT 10
    """)

    if sample:
        for r in sample:
            with st.expander(r.get("nombre", "Sin nombre")):
                st.text(r.get("descripcion", "Sin descripción"))
    else:
        st.info(f"No hay nodos de tipo {entity_type}")
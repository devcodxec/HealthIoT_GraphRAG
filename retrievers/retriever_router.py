from graph.neo4j_manager import Neo4jManager
from models.ollama_manager import OllamaManager
from retrievers.text2cypher import Text2CypherRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.manual_retriever import ManualRetriever

# Clase Orquestador de retrievers.
# Clasifica la pregunta del usuario y decide qué retriever usar:
# - structured → Text2Cypher (preguntas con conteos, listas, filtros)
# - semantic   → VectorRetriever (preguntas sobre conceptos o contexto)
# - manual     → ManualRetriever (preguntas frecuentes del dominio)
class RetrieverRouter:

    def __init__(self, neo4j: Neo4jManager):
        self.text2cypher = Text2CypherRetriever(neo4j)
        self.vector      = VectorRetriever(neo4j)
        self.manual      = ManualRetriever(neo4j)
        self.ai          = OllamaManager()

    # Clasifica la pregunta y la envía al retriever correcto.    
    def route(self, question: str) -> dict:        
        retriever_type = self._classify(question)
        print(f"  [Router] Tipo detectado: {retriever_type}")

        if retriever_type == "welcome":
            return self._handle_welcome(question)
        elif retriever_type == "manual":
            result = self._route_manual(question)
        elif retriever_type == "text2cypher":
            result = self.text2cypher.query(question)
        else:
            result = self.vector.query(question)

        # Garantiza que siempre existe 'type' en el resultado
        if "type" not in result:
            result["type"] = retriever_type

        return result

    # Clasifica la pregunta en uno de los tres tipos de consulta.
    # Usa palabras clave para tomar una decisión rápida sin llamar a la IA.
    def _classify(self, question: str) -> str:
        q = question.lower()

        welcome_keywords = [
            # Saludos - solo frases completas
            "hello", "hey",
            "hola", "buenos días", "buenas tardes", "buenas noches",
            "good morning", "good afternoon", "good evening",
            # Preguntas sobre el sistema
            "how are you", "cómo estás", "qué tal",
            "what are you", "who are you", "quién eres", "qué eres",
            # Agradecimientos
            "thank you", "gracias", "de nada",
            # Despedidas
            "goodbye", "adiós", "hasta luego",
        ]       

        # Primero verificar si es saludo o fuera del dominio
        if any(k in q for k in welcome_keywords):
            return "welcome"

        # Palabras clave que indican una consulta manual frecuente
        manual_keywords = [
            "most used", "most frequent", "most common",
            "más usado", "más frecuente", "más común",
            "list all devices", "listar dispositivos",
            "critical challenges", "desafíos críticos",
            "future work trends", "tendencias",
        ]
        if any(k in q for k in manual_keywords):
            return "manual"

        # Palabras clave que indican una consulta estructurada
        structured_keywords = [
            "how many", "cuántos", "count", "list", "listar",
            "which articles", "qué artículos", "filter",
            "which", "used more", "more than", 
            "order by", "top", "most", "least", "average",
            "what year", "qué año", "when", "cuándo",
        ]
        if any(k in q for k in structured_keywords):
            return "text2cypher"

        # Por defecto asumimos que es una consulta semántica
        return "vector"

    # Este método detecta qué consulta manual corresponde a la pregunta.
    def _route_manual(self, question: str) -> dict:
        
        q = question.lower()

        if "ecg" in q or "signal" in q or "señal" in q:
            return self.manual.devices_by_signal("ECG")

        if "wireless" in q or "technology" in q or "tecnolog" in q:
            return self.manual.most_used_technologies()

        if "ai model" in q or "modelo" in q or "algorithm" in q:
            return self.manual.most_used_ai_models()

        if "challenge" in q or "limitation" in q or "desafío" in q:
            return self.manual.critical_challenges()

        if "future" in q or "futuro" in q or "trend" in q:
            return self.manual.future_work_by_area()

        if "recommendation" in q or "recomendac" in q:
            return self.manual.recommendations_by_focus()

        if "edge" in q or "fog" in q or "cloud" in q:
            arch = "Edge" if "edge" in q else "Fog" if "fog" in q else "Cloud"
            return self.manual.articles_by_architecture(arch)

        # Si no coincide con ninguna consulta manual específica usamos Text2Cypher
        return self.text2cypher.query(question)
    
    # Este método maneja saludos y preguntas fuera del dominio 
    # respondiendo directamente con el LLM sin consultar Neo4j.
    def _handle_welcome(self, question: str) -> dict:
        prompt = (
            f"The user said: '{question}'\n\n"
            f"You are HealthIoT GraphRAG, an AI assistant specialized "
            f"in analyzing scientific literature about IoT health monitoring "
            f"systems. Respond naturally and briefly then invite the user "
            f"to ask questions about the corpus.\n\n"
            f"Examples of what you can help with:\n"
            f"- Devices used for ECG monitoring\n"
            f"- AI models for signal classification\n"
            f"- Wireless technologies in wearables\n"
            f"- Challenges and recommendations in IoT health\n\n"
            f"Keep your response under 3 sentences."
        )
        response = self.ai.chat(prompt)

        return {
            "type":     "welcome",
            "question": question,
            "results":  [],
            "answer":   response
        }
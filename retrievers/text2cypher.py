import logging
from graph.neo4j_manager import Neo4jManager
from models.ollama_manager import OllamaManager

# ─────────────────────────────────────────────────────────────
# ESQUEMA DEL GRAFO — describe los nodos y relaciones a la IA
# ─────────────────────────────────────────────────────────────
GRAPH_SCHEMA = """
NODE TYPES AND PROPERTIES:
- Article     {id, title, year, author, source_db, summary}
- Device      {name, brand, type, description}
- Signal      {name, description}
- Technology  {name, tech_type, description}
- Architecture{type, platform, layers, description}
- AIModel     {name, model_type, description}
- BigData     {name, status, tool, description}
- Challenge   {id, description, severity, barrier_type}
- Strength    {id, description, impact}
- Recommendation {id, description, focus}
- FutureWork  {id, description, area}
- Chunk       {id, text, section, embedding, hypo_questions}

RELATIONSHIPS:
(Article)-[:TIENE_FUENTE]->(Chunk)
(Article)-[:EMPLEA]->(Device)
(Article)-[:EMPLEA]->(Technology)
(Article)-[:EMPLEA]->(BigData)
(Article)-[:MONITORIZA]->(Signal)
(Article)-[:PROPONE]->(Architecture)
(Article)-[:UTILIZA]->(AIModel)
(Article)-[:IDENTIFICA]->(Challenge)
(Article)-[:IDENTIFICA]->(Strength)
(Article)-[:IDENTIFICA]->(Recommendation)
(Article)-[:DEFINE]->(FutureWork)
(Chunk)-[:MENCIONA]->(Device)
(Chunk)-[:MENCIONA]->(Signal)
(Chunk)-[:MENCIONA]->(Technology)
(Chunk)-[:MENCIONA]->(Architecture)
(Chunk)-[:MENCIONA]->(AIModel)
(Chunk)-[:MENCIONA]->(BigData)
(Chunk)-[:MENCIONA]->(Challenge)
(Chunk)-[:MENCIONA]->(Strength)
(Chunk)-[:MENCIONA]->(Recommendation)
(Chunk)-[:MENCIONA]->(FutureWork)
"""

# ─────────────────────────────────────────────────────────────
# MAPA TERMINOLÓGICO — traduce términos del usuario a nombres
# del grafo para evitar errores por sinónimos
# ─────────────────────────────────────────────────────────────
TERMINOLOGY_MAP = """
TERMINOLOGY MAP (user terms → graph values):
- "wearable", "sensor", "watch"     → Device.type
- "ECG", "heart rate", "SpO2"       → Signal.name
- "Bluetooth", "WiFi", "MQTT"       → Technology.name
- "Edge", "Fog", "Cloud"            → Architecture.type
- "CNN", "LSTM", "Random Forest"    → AIModel.name
- "limitation", "problem", "issue"  → Challenge.description
- "advantage", "contribution"       → Strength.description
- "future work", "next steps"       → FutureWork.description
- "wireless"                        → Technology.tech_type = 'Wireless'
- "protocol"                        → Technology.tech_type = 'Protocol'
- "classification", "detection"     → AIModel.model_type
"""

# ─────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES — ejemplos de pregunta → Cypher correctos
# Enseñan al modelo el formato esperado
# ─────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
EXAMPLES (question → Cypher):

Q: "List all devices that capture ECG signals"
A: MATCH (d:Device)<-[:MENCIONA]-(c:Chunk)-[:MENCIONA]->(s:Signal)
   WHERE s.name CONTAINS 'ECG'
   RETURN DISTINCT d.name, d.type
   LIMIT 25

Q: "Which wireless technologies are most used?"
A: MATCH (a:Article)-[:EMPLEA]->(t:Technology)
   WHERE t.tech_type = 'Wireless'
   RETURN t.name, count(a) AS frequency
   ORDER BY frequency DESC
   LIMIT 10

Q: "What AI models are used for classification?"
A: MATCH (a:Article)-[:UTILIZA]->(m:AIModel)
   WHERE m.model_type CONTAINS 'Classification'
   RETURN m.name, count(a) AS frequency
   ORDER BY frequency DESC
   LIMIT 10

Q: "What are the most critical challenges?"
A: MATCH (a:Article)-[:IDENTIFICA]->(ch:Challenge)
   WHERE ch.severity = 'Critical'
   RETURN ch.description, ch.barrier_type
   LIMIT 15

Q: "Which articles propose Edge architecture?"
A: MATCH (a:Article)-[:PROPONE]->(ar:Architecture)
   WHERE ar.type CONTAINS 'Edge'
   RETURN a.title, a.year, ar.platform
   ORDER BY a.year DESC
   LIMIT 10
"""

# ─────────────────────────────────────────────────────────────
# INSTRUCCIONES DE FORMATO — reglas estrictas para el Cypher
# ─────────────────────────────────────────────────────────────
FORMAT_INSTRUCTIONS = """
FORMAT INSTRUCTIONS:
- Return ONLY the Cypher query, no explanation, no markdown
- Always use LIMIT (max 25) to avoid large result sets
- Use CONTAINS for partial text matching, not exact =
- Use DISTINCT to avoid duplicate results
- Always RETURN meaningful properties, not entire nodes
- Use count() for frequency analysis
- For relationships use the exact names from the schema
"""

# Esta clase convierte preguntas en lenguaje natural a consultas Cypher usando el LLM.
# El prompt incluye los 4 elementos obligatorios del enunciado:
#    1. Esquema del modelo de datos (GRAPH_SCHEMA)
#    2. Instrucciones de formato (FORMAT_INSTRUCTIONS)
#    3. Mapa terminológico (TERMINOLOGY_MAP)
#    4. Few-shot examples (FEW_SHOT_EXAMPLES)

class Text2CypherRetriever:    

    MAX_RETRIES = 3  # Intentos de autocorrección si la query falla

    def __init__(self, neo4j: Neo4jManager):
        self.neo4j = neo4j
        self.ai    = OllamaManager()

    #Método principal: recibe una pregunta y devuelve los resultados del grafo.
    def query(self, question: str) -> dict:
        # Paso 1: generar la query Cypher
        cypher = self._generate_cypher(question)
        print(f"  [Text2Cypher] Query generada:\n  {cypher}")

        # Paso 2: ejecutar con autocorrección
        for attempt in range(self.MAX_RETRIES):
            try:
                results = self.neo4j.run_query(cypher)
                return {
                    "type":    "text2cypher",
                    "question": question,
                    "cypher":  cypher,
                    "results": results
                }
            except Exception as error:
                print(f"  [Text2Cypher] Error intento {attempt+1}: {error}")
                if attempt < self.MAX_RETRIES - 1:
                    # Autocorrección: reintenta con el error en el prompt
                    cypher = self._correct_cypher(cypher, str(error), question)
                    print(f"  [Text2Cypher] Query corregida:\n  {cypher}")

        return {"type": "text2cypher", "question": question,
                "cypher": cypher, "results": [], "error": "Max retries reached"}

    # Genera la query Cypher a partir de la pregunta.
    def _generate_cypher(self, question: str) -> str:        
        prompt = (
            f"Convert this question to a Cypher query.\n\n"
            f"GRAPH SCHEMA:\n{GRAPH_SCHEMA}\n\n"
            f"TERMINOLOGY MAP:\n{TERMINOLOGY_MAP}\n\n"
            f"FORMAT INSTRUCTIONS:\n{FORMAT_INSTRUCTIONS}\n\n"
            f"EXAMPLES:\n{FEW_SHOT_EXAMPLES}\n\n"
            f"Question: {question}\n"
            f"Cypher query:"
        )
        response = self.ai.chat(prompt)
        return self._clean_cypher(response)

    # Autocorrección: cuando la query falla, reintenta incluyendo el mensaje de error en el prompt.
    def _correct_cypher(self, bad_cypher: str,
                        error: str, question: str) -> str:
        prompt = (
            f"This Cypher query failed:\n{bad_cypher}\n\n"
            f"Error: {error}\n\n"
            f"Original question: {question}\n\n"
            f"GRAPH SCHEMA:\n{GRAPH_SCHEMA}\n\n"
            f"Fix the query. Return ONLY the corrected Cypher:"
        )
        response = self.ai.chat(prompt)
        return self._clean_cypher(response)

    # Limpia la respuesta del modelo eliminando markdown.
    def _clean_cypher(self, response: str) -> str:        
        clean = response.strip()
        clean = clean.replace("```cypher", "").replace("```", "")
        # Se queda solo con la primera query si devuelve varias
        lines = [l for l in clean.split("\n") if l.strip()]
        return "\n".join(lines).strip()
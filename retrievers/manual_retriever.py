from graph.neo4j_manager import Neo4jManager

 # Clase que contiene consultas Cypher manuales para preguntas frecuentes y complejas del dominio IoT salud.
 # Cuando se usa? Cuando la pregunta es frecuente, compleja y siempre tiene la misma estructura. 
 # En lugar de generar el Cypher con el LLM en cada consulta, implementamos consultas manuales directamente en el código.

class ManualRetriever:

    def __init__(self, neo4j: Neo4jManager):
        self.neo4j = neo4j

    # ──────────────────────────────────────────
    # CONSULTAS ESTRUCTURADAS DEL ENUNCIADO
    # ──────────────────────────────────────────

    # Qué dispositivos capturan una señal específica?
    # Ejemplo: "Lista todos los dispositivos que capturan ECG"      
    def devices_by_signal(self, signal_name: str) -> dict:
        
        query = """
        MATCH (d:Device)<-[:MENCIONA]-(c:Chunk)-[:MENCIONA]->(s:Signal)
        WHERE s.name CONTAINS $signal
        RETURN DISTINCT d.name AS device,
                        d.type AS type,
                        s.name AS signal
        ORDER BY d.name
        """
        results = self.neo4j.run_query(query, {"signal": signal_name})
        return {"query": "devices_by_signal",
                "params": {"signal": signal_name},
                "results": results}

    # Qué tecnologías inalámbricas se usan más?
    # Ejemplo: "Cuáles son las tecnologías más utilizadas"    
    def most_used_technologies(self, tech_type: str = None) -> dict:
        if tech_type:
            query = """
            MATCH (a:Article)-[:EMPLEA]->(t:Technology)
            WHERE t.tech_type = $tech_type
            RETURN t.name AS technology,
                   t.tech_type AS type,
                   count(a) AS frequency
            ORDER BY frequency DESC
            LIMIT 15
            """
            params = {"tech_type": tech_type}
        else:
            query = """
            MATCH (a:Article)-[:EMPLEA]->(t:Technology)
            RETURN t.name AS technology,
                   t.tech_type AS type,
                   count(a) AS frequency
            ORDER BY frequency DESC
            LIMIT 15
            """
            params = {}

        results = self.neo4j.run_query(query, params)
        return {"query": "most_used_technologies",
                "params": params, "results": results}

    
    # Qué modelos de IA son más mencionados?
    # Ejemplo: "Qué tecnologías de desarrollo son más mencionadas"    
    def most_used_ai_models(self) -> dict:        
        query = """
        MATCH (a:Article)-[:UTILIZA]->(m:AIModel)
        RETURN m.name AS model,
               m.model_type AS type,
               count(a) AS frequency
        ORDER BY frequency DESC
        LIMIT 15
        """
        results = self.neo4j.run_query(query, {})
        return {"query": "most_used_ai_models", "results": results}

    # Cuáles son los desafíos más críticos?
    # Ejemplo: "Cuáles son los desafíos que más se presentan"
    def critical_challenges(self) -> dict:
        query = """
        MATCH (a:Article)-[:IDENTIFICA]->(ch:Challenge)
        RETURN ch.description AS challenge,
               ch.severity AS severity,
               ch.barrier_type AS type,
               count(a) AS frequency
        ORDER BY
            CASE ch.severity
                WHEN 'Critical' THEN 1
                WHEN 'Moderate' THEN 2
                ELSE 3
            END,
            frequency DESC
        LIMIT 20
        """
        results = self.neo4j.run_query(query, {})
        return {"query": "critical_challenges", "results": results}

    # Qué artículos proponen una arquitectura específica?
    # Ejemplo: "Artículos que proponen arquitectura Edge"     
    def articles_by_architecture(self, arch_type: str) -> dict:        
        query = """
        MATCH (a:Article)-[:PROPONE]->(ar:Architecture)
        WHERE ar.type CONTAINS $arch_type
        RETURN a.title AS title,
               a.year AS year,
               ar.type AS architecture,
               ar.platform AS platform
        ORDER BY a.year DESC
        LIMIT 15
        """
        results = self.neo4j.run_query(
            query, {"arch_type": arch_type}
        )
        return {"query": "articles_by_architecture",
                "params": {"arch_type": arch_type},
                "results": results}
    
    # Qué tendencias de trabajo futuro hay?
    # Ejemplo: "Qué tendencias de trabajo futuro se repiten"    
    def future_work_by_area(self, area: str = None) -> dict:
        
        if area:
            query = """
            MATCH (a:Article)-[:DEFINE]->(fw:FutureWork)
            WHERE fw.area = $area
            RETURN fw.description AS future_work,
                   fw.area AS area,
                   a.title AS article
            LIMIT 20
            """
            params = {"area": area}
        else:
            query = """
            MATCH (a:Article)-[:DEFINE]->(fw:FutureWork)
            RETURN fw.area AS area,
                   count(fw) AS frequency,
                   collect(fw.description)[0..3] AS examples
            ORDER BY frequency DESC
            """
            params = {}

        results = self.neo4j.run_query(query, params)
        return {"query": "future_work_by_area",
                "params": params, "results": results}


    # Qué recomendaciones técnicas hay?
    # Ejemplo: "Qué recomendaciones ofrecen los autores"        
    def recommendations_by_focus(self, focus: str = None) -> dict:        
        if focus:
            query = """
            MATCH (a:Article)-[:IDENTIFICA]->(r:Recommendation)
            WHERE r.focus = $focus
            RETURN r.description AS recommendation,
                   r.focus AS focus,
                   a.title AS article
            LIMIT 15
            """
            params = {"focus": focus}
        else:
            query = """
            MATCH (a:Article)-[:IDENTIFICA]->(r:Recommendation)
            RETURN r.focus AS focus,
                   count(r) AS frequency,
                   collect(r.description)[0..3] AS examples
            ORDER BY frequency DESC
            """
            params = {}

        results = self.neo4j.run_query(query, params)
        return {"query": "recommendations_by_focus",
                "params": params, "results": results}
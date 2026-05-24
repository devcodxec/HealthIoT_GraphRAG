import uuid
from typing import Dict, Any, List
from graph.neo4j_manager import Neo4jManager

# Clase Responsable de escribir entidades en Neo4j.
# Recibe el diccionario que devuelve EntityExtractor.extract_all() y crea los nodos y relaciones correspondientes.
# Usa MERGE en lugar de CREATE para que si el mismo Device o Signal aparece en varios artículos no se duplique el nodo,
# sino que se reutilice y se le añadan nuevas relaciones.
class GraphService:
    

    def __init__(self, db: Neo4jManager):
        self.db = db

    # Crea o actualiza el nodo Article raíz del artículo.
    def save_article(self, article_id: str, metadata: Dict[str, Any]):
        
        query = """
        MERGE (a:Article {id: $id})
        SET a.title      = $title,
            a.year       = $year,
            a.author     = $author,
            a.summary  = $summary,
            a.conclusions    = $conclusions
        """
        self.db.run_query(query, {
            "id":        article_id,
            "title":     metadata.get("title", ""),
            "year":      metadata.get("year"),
            "author":    metadata.get("author", ""),
            "summary": metadata.get("summary", ""),
            "conclusions":   metadata.get("conclusions", ""),
        })

     # Crea el nodo Chunk y lo conecta al Article.
     # Guarda el embedding y las preguntas hipotéticas para la búsqueda semántica posterior.
    def save_chunk(self, chunk_id: str, text: str, section: str,
                   index: int, article_id: str,
                   embedding: List[float], hypo_questions: List[str]):
       
        query = """
        MATCH (a:Article {id: $article_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text           = $text,
            c.section        = $section,
            c.index          = $index,
            c.embedding      = $embedding,
            c.hypo_questions = $hypo_questions
        MERGE (a)-[:TIENE_FUENTE]->(c)
        """
        self.db.run_query(query, {
            "chunk_id":       chunk_id,
            "text":           text,
            "section":        section,
            "index":          index,
            "embedding":      embedding,
            "hypo_questions": hypo_questions,
            "article_id":     article_id,
        })

    # Guarda todos los tipos de entidad extraídos en las 3 consultas.
    # Cada tipo tiene su propia etiqueta Neo4j y su relación con Article.
    def save_typed_entities(self, entities: Dict[str, Any],
                            article_id: str, chunk_id: str):
       
        self._save_devices(entities.get("devices", []), article_id, chunk_id)
        self._save_signals(entities.get("signals", []), article_id, chunk_id)
        self._save_technologies(entities.get("technologies", []), article_id, chunk_id)
        self._save_architectures(entities.get("architectures", []), article_id, chunk_id)
        self._save_ai_models(entities.get("ai_models", []), article_id, chunk_id)
        self._save_bigdata(entities.get("bigdata", []), article_id, chunk_id)
        self._save_challenges(entities.get("challenges", []), article_id, chunk_id)
        self._save_strengths(entities.get("strengths", []), article_id, chunk_id)
        self._save_recommendations(entities.get("recommendations", []), article_id, chunk_id)
        self._save_future_works(entities.get("future_works", []), article_id, chunk_id)

    # ── Métodos privados por tipo de entidad ──────────────────────────────

    def _save_devices(self, devices: list, article_id: str, chunk_id: str):
        for d in devices:
            if not d.get("name"):
                continue
            # MERGE evita duplicados: si ya existe Device {name: 'Apple Watch'}
            # no lo crea de nuevo, solo añade la relación EMPLEA
            query = """
            MERGE (dev:Device {name: $name})
            SET dev.brand = $brand, dev.type = $type, dev.description = $description
            WITH dev
            MATCH (a:Article {id: $article_id})
            MERGE (a)-[:EMPLEA]->(dev)
            WITH dev
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENCIONA]->(dev)
            """
            self.db.run_query(query, {
                "name":       d.get("name", ""),
                "brand":      d.get("brand", ""),
                "type":       d.get("type", "Other"),
                "description": d.get("description", ""),
                "article_id": article_id,
                "chunk_id":   chunk_id,
            })

    def _save_signals(self, signals: list, article_id: str, chunk_id: str):
        for s in signals:
            if not s.get("name"):
                continue
            query = """
            MERGE (sig:Signal {name: $name})
            SET sig.description = $description
            WITH sig
            MATCH (a:Article {id: $article_id})
            MERGE (a)-[:MONITORIZA]->(sig)
            WITH sig
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENCIONA]->(sig)
            """
            self.db.run_query(query, {
                "name":       s.get("name", ""),
                "description": s.get("description", ""),
                "article_id": article_id,
                "chunk_id":   chunk_id,
            })

    def _save_technologies(self, techs: list, article_id: str, chunk_id: str):
        for t in techs:
            if not t.get("name"):
                continue
            query = """
            MERGE (tech:Technology {name: $name})
            SET tech.tech_type = $tech_type, tech.description = $description
            WITH tech
            MATCH (a:Article {id: $article_id})
            MERGE (a)-[:EMPLEA]->(tech)
            WITH tech
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENCIONA]->(tech)
            """
            self.db.run_query(query, {
                "name":       t.get("name", ""),
                "tech_type":  t.get("tech_type", "Other"),
                "description": t.get("description", ""),
                "article_id": article_id,
                "chunk_id":   chunk_id,
            })

    def _save_architectures(self, archs: list, article_id: str, chunk_id: str):
        for arch in archs:
            if not arch.get("type"):
                continue
            query = """
            MERGE (ar:Architecture {type: $type})
            SET ar.platform = $platform, ar.layers = $layers, ar.description = $description
            WITH ar
            MATCH (a:Article {id: $article_id})
            MERGE (a)-[:PROPONE]->(ar)
            WITH ar
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENCIONA]->(ar)
            """
            self.db.run_query(query, {
                "type":       arch.get("type", "Other"),
                "platform":   arch.get("platform", ""),
                "layers":     arch.get("layers", ""),
                "description": arch.get("description", ""),
                "article_id": article_id,
                "chunk_id":   chunk_id,
            })

    def _save_ai_models(self, models: list, article_id: str, chunk_id: str):
        for m in models:
            if not m.get("name"):
                continue
            query = """
            MERGE (ai:AIModel {name: $name})
            SET ai.model_type = $model_type, ai.description = $description
            WITH ai
            MATCH (a:Article {id: $article_id})
            MERGE (a)-[:UTILIZA]->(ai)
            WITH ai
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENCIONA]->(ai)
            """
            self.db.run_query(query, {
                "name":       m.get("name", ""),
                "model_type": m.get("model_type", "Other"),
                "description": m.get("description", ""),
                "article_id": article_id,
                "chunk_id":   chunk_id,
            })

    def _save_bigdata(self, bigdata: list, article_id: str, chunk_id: str):
        for b in bigdata:
            if not b.get("name"):
                continue
            query = """
            MERGE (bd:BigData {name: $name})
            SET bd.status = $status, bd.tool = $tool, bd.description = $description
            WITH bd
            MATCH (a:Article {id: $article_id})
            MERGE (a)-[:EMPLEA]->(bd)
            WITH bd
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENCIONA]->(bd)
            """
            self.db.run_query(query, {
                "name":       b.get("name", ""),
                "status":     b.get("status", "Proposed"),
                "tool":       b.get("tool", ""),
                "description": b.get("description", ""),
                "article_id": article_id,
                "chunk_id":   chunk_id,
            })

    def _save_challenges(self, challenges: list, article_id: str, chunk_id: str):
        for ch in challenges:
            if not ch.get("description"):
                continue
            # Challenge no tiene nombre único, usamos UUID para no mezclar
            query = """
            MATCH (a:Article {id: $article_id})
            MATCH (c:Chunk {id: $chunk_id})
            CREATE (ch:Challenge {
                id:           randomUUID(),
                description:  $description,
                severity:     $severity,
                barrier_type: $barrier_type
            })
            MERGE (a)-[:IDENTIFICA]->(ch)
            MERGE (c)-[:MENCIONA]->(ch)
            """
            self.db.run_query(query, {
                "description":  ch.get("description", ""),
                "severity":     ch.get("severity", "Moderate"),
                "barrier_type": ch.get("barrier_type", "Technical"),
                "article_id":   article_id,
                "chunk_id":     chunk_id,
            })

    def _save_strengths(self, strengths: list, article_id: str, chunk_id: str):
        for s in strengths:
            if not s.get("description"):
                continue
            query = """
            MATCH (a:Article {id: $article_id})
            MATCH (c:Chunk {id: $chunk_id})
            CREATE (st:Strength {
                id:          randomUUID(),
                description: $description,
                impact:      $impact
            })
            MERGE (a)-[:IDENTIFICA]->(st)
            MERGE (c)-[:MENCIONA]->(st)
            """
            self.db.run_query(query, {
                "description": s.get("description", ""),
                "impact":      s.get("impact", "Medium"),
                "article_id":  article_id,
                "chunk_id":    chunk_id,
            })

    def _save_recommendations(self, recs: list, article_id: str, chunk_id: str):
        for r in recs:
            if not r.get("description"):
                continue
            query = """
            MATCH (a:Article {id: $article_id})
            MATCH (c:Chunk {id: $chunk_id})
            CREATE (rc:Recommendation {
                id:          randomUUID(),
                description: $description,
                focus:       $focus
            })
            MERGE (a)-[:IDENTIFICA]->(rc)
            MERGE (c)-[:MENCIONA]->(rc)
            """
            self.db.run_query(query, {
                "description": r.get("description", ""),
                "focus":       r.get("focus", "Algorithm"),
                "article_id":  article_id,
                "chunk_id":    chunk_id,
            })

    def _save_future_works(self, fws: list, article_id: str, chunk_id: str):
        for fw in fws:
            if not fw.get("description"):
                continue
            query = """
            MATCH (a:Article {id: $article_id})
            MATCH (c:Chunk {id: $chunk_id})
            CREATE (f:FutureWork {
                id:          randomUUID(),
                description: $description,
                area:        $area
            })
            MERGE (a)-[:DEFINE]->(f)
            MERGE (c)-[:MENCIONA]->(f)
            """
            self.db.run_query(query, {
                "description": fw.get("description", ""),
                "area":        fw.get("area", "AI"),
                "article_id":  article_id,
                "chunk_id":    chunk_id,
            })
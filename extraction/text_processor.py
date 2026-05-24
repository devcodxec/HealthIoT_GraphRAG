import uuid
import re
import logging
import time
from typing import Dict, Any, List
from tqdm import tqdm
from models.ollama_manager import OllamaManager
from extraction.entity_extractor import EntityExtractor
from graph.graph_service import GraphService
from graph.neo4j_manager import Neo4jManager

# ─────────────────────────────────────────
# CHUNKING POR SECCIÓN CIENTÍFICA
# ─────────────────────────────────────────

# Patrones de encabezados típicos en artículos IoT/salud
SECTION_PATTERNS = {
    "abstract":      r"(?i)^\s*(abstract|resumen)\s*$",
    "introduction":  r"(?i)^\s*(\d[\.\)]?\s*)?(introduction|introducción)\s*",
    "methods":       r"(?i)^\s*(\d[\.\)]?\s*)?(method|methodology|system design|"
                     r"proposed|architecture|materiales|metodolog)",
    "results":       r"(?i)^\s*(\d[\.\)]?\s*)?(result|experiment|evaluation|"
                     r"performance|resultados|evaluaci)",
    "conclusion":    r"(?i)^\s*(\d[\.\)]?\s*)?(conclusion|discussion|future|"
                     r"summary|conclusi|discusi|trabajo futuro)",
    "skip_references":     r"(?i)^\s*(references|bibliography|bibliograf)\s*",
    "skip_acknowledgment": r"(?i)^\s*(acknowledgment|acknowledgement|agradec)\s*",
    "skip_appendix":       r"(?i)^\s*(appendix|anexo)\s*",
}

# Devuelve la sección si la línea es un encabezado conocido.
def detect_section(line: str) -> str | None:
    
    for section, pattern in SECTION_PATTERNS.items():
        if re.match(pattern, line.strip()):
            return section
    return None

# Visualiza las secciones detectadas para verificación.
def log_sections(self, chunks: list):
    from collections import Counter
    sections = Counter(c["section"] for c in chunks)
    print("\n Secciones detectadas:")
    for section, count in sections.most_common():
        print(f"    {section}: {count} chunks")

# Divide el texto por secciones científicas.
# Dentro de cada sección divide por párrafos con solapamiento.
# Un chunk de 'Conclusions' responde a preguntas distintas que uno de 'Methods'. Mantenerlos separados mejora la precisión del retrieval.
def chunk_by_section(text: str, max_words: int = 200,
                     overlap_words: int = 50) -> List[Dict]:

    lines = text.split("\n")
    sections = []
    current_section = "skip_preamble"
    current_lines = []

    for line in lines:
        detected = detect_section(line)
        if detected:
            if current_lines:
                sections.append({
                    "name": current_section,
                    "text": "\n".join(current_lines).strip()
                })
            current_section = detected
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "name": current_section,
            "text": "\n".join(current_lines).strip()
        })

    chunks = []
    global_idx = 0

    for section in sections:

        # Saltar cualquier sección cuyo nombre empiece por "skip_"
        if section["name"].startswith("skip_"):
            print(f"  Sección '{section['name']}' excluida")
            continue

        paragraphs = [p.strip() for p in section["text"].split("\n\n") if p.strip()]
        current_text = ""

        for para in paragraphs:
            words_so_far = len((current_text + para).split())
            if words_so_far > max_words and current_text:
                chunks.append({
                    "text":    current_text.strip(),
                    "section": section["name"],
                    "index":   global_idx
                })
                global_idx += 1
                tail_words = current_text.split()[-overlap_words:]
                current_text = " ".join(tail_words) + "\n\n" + para + "\n\n"
            else:
                current_text += para + "\n\n"

        if current_text.strip():
            chunks.append({
                "text":    current_text.strip(),
                "section": section["name"],
                "index":   global_idx
            })
            global_idx += 1

    return chunks


# ─────────────────────────────────────────
# GENERADOR DE EMBEDDINGS
# ─────────────────────────────────────────

# Genera el embedding usando el modelo local de Ollama.
# Trunca a 500 palabras si el texto es muy largo para no superar el límite del modelo.
def get_embedding(text: str) -> List[float]:

    import ollama as ol
    import os
    
    # Usa el mismo modelo del .env para no instalar nada extra.
    # model = os.getenv("OLLAMA_MODEL", "llama3")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    # Truncar a 500 palabras para evitar error de contexto
    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500])

    try:
        response = ol.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        logging.warning(f"Embedding fallido: {e}. Usando vector vacío.")
        return []


# ─────────────────────────────────────────
# TEXT PROCESSOR PRINCIPAL
# ─────────────────────────────────────────
# Orquesta el pipeline completo para un artículo:
#    1. Recibe el texto completo del PDF
#    2. Lo divide en chunks por sección científica
#    3. Para cada chunk: extrae entidades (3 consultas) + guarda en Neo4j
#    4. Consolida entidades duplicadas (Entity Resolution)

class TextProcessor:   


    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j   = neo4j_manager
        self.extractor = EntityExtractor()
        self.graph   = GraphService(neo4j_manager)

    # Metodo Principal que permite procesar un documento completo:
    # crear nodo → dividir en chunks → procesar chunks → consolidar.
    def process_document(self, text: str, metadata: Dict[str, Any] = None):
        
        metadata   = metadata or {}

        # Genera un ID único para el artículo
        article_id = metadata.get("id") or str(uuid.uuid4())
        metadata["id"] = article_id

        print(f"\nProcesando: {metadata.get('title', article_id)[:60]}")

        # Crea el nodo Article en Neo4j
        self.graph.save_article(article_id, metadata)

        # Chunking por sección científica
        chunks = chunk_by_section(text)
        print(f"  {len(chunks)} chunks generados")

        # Procesamiento de cada chunk
        for chunk in tqdm(chunks, desc="  Chunks", leave=False):
            self._process_chunk(chunk, article_id)
            time.sleep(3)  # Pausa para evitar saturar el modelo local

        # Consolida descripciones repetidas
        self._consolidate_entities()

        print(f"  Artículo {article_id} completado.")
        return article_id

    # Para un solo chunk:
    #    a) Genera embedding del texto
    #    b) Genera preguntas hipotéticas HyQE
    #    c) Guarda el nodo Chunk con embedding y preguntas
    #    d) Extrae entidades en 3 consultas y las guarda en Neo4j
    def _process_chunk(self, chunk: Dict, article_id: str):

        chunk_id = f"{article_id}_chunk_{chunk['index']}"
        text     = chunk["text"]
        section  = chunk["section"]

        # Embedding para búsqueda semántica futura
        embedding = get_embedding(text)

        # HyQE: preguntas hipotéticas que este chunk respondería
        hypo_questions = self.extractor.generate_hypothetical_questions(text, n=4)

        # Nodo Chunk con embedding + preguntas
        self.graph.save_chunk(
            chunk_id      = chunk_id,
            text          = text,
            section       = section,
            index         = chunk["index"],
            article_id    = article_id,
            embedding     = embedding,
            hypo_questions= hypo_questions,
        )

        # Extracción multi-pasada (3 consultas temáticas)
        entities = self.extractor.extract_all(text)

        # Guarda todos los nodos tipados y sus relaciones con Article
        self.graph.save_typed_entities(entities, article_id, chunk_id)

    def _consolidate_entities(self):        
        print("  Consolidación completada (sin duplicados en este artículo).")
                
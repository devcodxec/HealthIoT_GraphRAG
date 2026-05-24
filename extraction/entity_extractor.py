import logging
from typing import Tuple, List, Dict, Any
from models.ollama_manager import OllamaManager
from extraction.schema import HardwareResult, AnalyticsResult, ImpactResult, EntityClassification

# ─────────────────────────────────────────────
# DICCIONARIO DE SINÓNIMOS DEL DOMINIO
# Estandariza nombres antes de guardar en Neo4j
# ─────────────────────────────────────────────
SYNONYMS = {
    # Señales
    "ELECTROCARDIOGRAM": "ECG",
    "ELECTROCARDIOGRAMA": "ECG",
    "EKG": "ECG",
    "HEART RATE": "Heart Rate",
    "PULSE RATE": "Heart Rate",
    "HR": "Heart Rate",
    "OXYGEN SATURATION": "SpO2",
    "BLOOD OXYGEN": "SpO2",
    "SPO2": "SpO2",
    # Tecnologías
    "BLE": "Bluetooth Low Energy",
    "BLUETOOTH LE": "Bluetooth Low Energy",
    "BLUETOOTH 5.0": "Bluetooth Low Energy",
    "LORAWAN": "LoRa",
    "NARROWBAND IOT": "NB-IoT",
    "MESSAGE QUEUING TELEMETRY TRANSPORT": "MQTT",
    # Modelos IA
    "CONVOLUTIONAL NEURAL NETWORK": "CNN",
    "LONG SHORT-TERM MEMORY": "LSTM",
    "RANDOM FOREST CLASSIFIER": "Random Forest",
    "CONVOLUTIONAL NEURAL NETWORKS (CNN)": "CNN",
    "RECURRENT NEURAL NETWORKS (RNN)": "RNN",
    "LONG SHORT-TERM MEMORY (LSTM)": "LSTM",
    "LONG SHORT-TERM MEMORY (LSTM)": "LSTM",
    "SUPPORT VECTOR MACHINES (SVM)": "SVM",
    # Dispositivos
    "SMART WATCH": "Smartwatch",
    "IWATCH": "Apple Watch",
    "APPLE WATCH SERIES": "Apple Watch",
}

VALID_ENTITY_TYPES = {
    "Device", "Signal", "Technology", "Architecture",
    "AIModel", "BigData", "Challenge", "Strength",
    "Recommendation", "FutureWork"
}

VALID_SIGNALS = {
    "ECG", "EKG", "ELECTROCARDIOGRAM",
    "SPO2", "OXYGEN SATURATION", "BLOOD OXYGEN",
    "HEART RATE", "HR", "PULSE RATE",
    "BLOOD PRESSURE", "BP",
    "TEMPERATURE", "BODY TEMPERATURE",
    "EEG", "ELECTROENCEPHALOGRAM",
    "EMG", "ELECTROMYOGRAM",
    "RESPIRATORY RATE", "BREATHING RATE",
    "GLUCOSE", "BLOOD GLUCOSE",
    "ACCELEROMETER", "GYROSCOPE",
    "GALVANIC SKIN RESPONSE", "GSR"
}

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Clase EntityExtractor
# Define métodos para extraer entidades del dominio IoT salud en diferentes consultas y normaliza nombres usando el diccionario de sinónimos. 
# Al separar en diferentes consultas, el modelo se concentra mejor en cada tipo de entidad mejorando el recall.
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


class EntityExtractor:


    def __init__(self):
        self.ai = OllamaManager()

    def normalize(self, name: str) -> str:
        """
        Estandariza un nombre usando el diccionario de sinónimos.
        Ejemplo: 'BLE' → 'Bluetooth Low Energy'
        """
        if not name:
            return "UNKNOWN"
        key = name.strip().upper()
        return SYNONYMS.get(key, name.strip())

    # CONSULTA 1: Hardware e infraestructura
    # Extrae: Device, Signal, Technology, Architecture.
    # Son los componentes físicos y de red del sistema IoT.
    def extract_hardware(self, text: str) -> HardwareResult:
        system_prompt = (
            "Extract IoT health monitoring entities from scientific text. "
            "Return valid JSON only."
        )
        return self.ai.structured_output(
            prompt=(
                "From this text, extract:\n"
                "- devices: wearable devices (smartwatch, bracelet, patch, sensor)\n"
                "- signals: body signals (ECG, heart rate, SpO2, temperature)\n"
                "- technologies: protocols (Bluetooth, MQTT, WiFi, Zigbee, LoRa)\n"
                "- architectures: system types (Edge, Fog, Cloud)\n\n"
                f"Text:\n{text}"
            ),
            schema=HardwareResult,
            system_prompt=system_prompt,
        )

    # CONSULTA 2: Analítica e IA
    # Extrae: AIModel, BigData.
    # Separa la consulta 1 para que el modelo se concentre en terminología ML/DL y Big Data exclusivamente mejorando el recall de estas entidades.
    def extract_analytics(self, text: str) -> AnalyticsResult:       
        system_prompt = (
        "You are an AI and Big Data expert for healthcare IoT. "
        "Extract ONLY specific AI models and Big Data tools. "
        "For AIModel: ONLY specific algorithms "
        "(CNN, LSTM, Random Forest, SVM, XGBoost, Transformer). "
        "NOT generic terms like 'machine learning' or 'deep learning'. "
        "For BigData: ONLY specific tools or techniques "
        "(Hadoop, Spark, Kafka, Flink, streaming, batch processing). "
        "NOT generic terms like 'data analytics' or 'data processing'. "
        "If no specific entity is found, return empty lists."
        )
        return self.ai.structured_output(
            prompt=f"Extract specific AI and Big Data entities from:\n\n{text}",
            schema=AnalyticsResult,
            system_prompt=system_prompt,
        )
  
    # CONSULTA 3: Impacto cualitativo
    # Extrae: Challenge, Strength, Recommendation, FutureWork.
    # Son los mas omitidos en una consulta única porque requieren razonamiento sobre el texto, no solo reconocimiento de nombres.
    def extract_impact(self, text: str) -> ImpactResult:
        system_prompt = (
            "You are a systematic literature review expert for IoT health systems. "
            "Extract ONLY qualitative findings: "
            "technical/ethical/privacy challenges and their severity, "
            "system strengths and their impact level, "
            "technical recommendations with their focus area, "
            "and future work directions proposed by the authors."
        )
        return self.ai.structured_output(
            prompt=f"Extract challenges, strengths, recommendations and future work from:\n\n{text}",
            schema=ImpactResult,
            system_prompt=system_prompt,
        )

    # MÉTODO UNIFICADO DE EXTRACCIÓN
    # Extrae todas las entidades del texto en consultas selectivas.
    # Flujo:
    # 1. Clasificación previa → detecta qué tipos hay en el texto.
    # 2. Solo llama a los extractores relevantes.
    # 3. Si ningún extractor encuentra nada → reintento.
    # 4. Normaliza nombres y devuelve diccionario unificado.
    def extract_all(self, text: str) -> Dict[str, Any]:

        # Paso 1: clasificación previa
        entity_types = self.classify_entities_present(text)
        print(f"    Tipos detectados: {entity_types}")

        # Paso 2: llamar solo a los extractores relevantes
        hardware_types  = {"Device", "Signal", "Technology", "Architecture"}
        analytics_types = {"AIModel", "BigData"}
        impact_types    = {"Challenge", "Strength", "Recommendation", "FutureWork"}

        detected  = set(entity_types)
        hardware  = HardwareResult()
        analytics = AnalyticsResult()
        impact    = ImpactResult()

        if detected & hardware_types:
            print("    → Extrayendo hardware...")
            hardware = self.extract_hardware(text)

        if detected & analytics_types:
            print("    → Extrayendo analítica...")
            analytics = self.extract_analytics(text)

        if detected & impact_types:
            print("    → Extrayendo impacto...")
            impact = self.extract_impact(text)

        # Paso 3: reintento si no se encontró nada
        total_found = (
            len(hardware.devices)    +
            len(hardware.signals)    +
            len(analytics.ai_models) +
            len(impact.challenges)
        )

        if total_found == 0:
            logging.warning("Recall cero, reintentando con prompt más explícito...")
            hardware = self.ai.structured_output(
                prompt=(
                    "This is an IoT health monitoring paper. "
                    "Even if entities appear implicit, identify ANY "
                    f"device, signal or technology mentioned:\n\n{text}"
                ),
                schema=HardwareResult,
                system_prompt="Extract all possible IoT health entities.",
            )

        # Paso 4: normalizar y devolver
        result = {
            "devices":         self._normalize_list(hardware.devices, "name"),
            "signals":         self._normalize_list(hardware.signals, "name"),
            "technologies":    self._normalize_list(hardware.technologies, "name"),
            "architectures":   [a.model_dump() for a in hardware.architectures],
            "ai_models":       self._normalize_list(analytics.ai_models, "name"),
            "bigdata":         self._normalize_list(analytics.bigdata, "name"),
            "challenges":      [c.model_dump() for c in impact.challenges],
            "strengths":       [s.model_dump() for s in impact.strengths],
            "recommendations": [r.model_dump() for r in impact.recommendations],
            "future_works":    [f.model_dump() for f in impact.future_works],
        }

        return result

    # Aplica normalize() al campo indicado de cada item de la lista.
    def _normalize_list(self, items: list, field: str) -> List[Dict]:
        
        normalized = []
        for item in items:
            d = item.model_dump()
            if field in d and d[field]:
                d[field] = self.normalize(d[field])
                # Filtrar señales no válidas
                if hasattr(item, 'name') and 'signal' in str(type(item)).lower():
                    if d[field].upper() not in VALID_SIGNALS:
                        continue
            normalized.append(d)
        return normalized

    
    # CONSOLIDACIÓN (Entity Resolution)
    # Cuando la misma entidad aparece en varios chunks con descripciones distintas, fusionamos todo en un único resumen coherente.    
    def summarize_entity(self, name: str, descriptions: List[str]) -> str:

        # Si no hay descripciones no hay nada que resumir
        if not descriptions:
            return ""
        
        # Une todas las descripciones en un solo texto separadas por " | " para que la IA las distinga.
        joined = " | ".join(descriptions)

        # Pedimos a la IA que haga un resumen de todo.
        return self.ai.chat(
            f"Create a single concise technical summary about '{name}' "
            f"based on these descriptions from different papers:\n{joined}"
        )

    # Resume múltiples descripciones de la misma relación entre dos entidades.
    # Si 5 artículos dicen que "Apple Watch captura ECG" cada uno lo explica diferente. Este método fusiona todas esas descripciones en un resumen de la relación.
    def summarize_relationship(self, source: str, target: str,
                                descriptions: List[str]) -> str:
       
        # Si no hay descripciones devuelve vacío.
        if not descriptions:
            return ""
        
        # Une todas las descripciones
        joined = " | ".join(descriptions)

        # Pedimos el resumen de la relación entre dos entidades
        return self.ai.chat(
            f"Summarize the relationship between '{source}' and '{target}' "
            f"based on: {joined}"
        )

    # HyQE (Hypothetical Question Embeddings):
    # Genera  preguntas hipotéticas que un fragmento de texto respondería.
    # Es útil cuando hay una brecha entre cómo pregunta un investigador ('Qué tecnologías inalámbricas se usan?') y cómo escribe el artículo ('se implementó un sistema de transmisión mediante Bluetooth 5.0'). 
    # Guardar preguntas hipotéticas en el Chunk permite recuperarlo buscando por la pregunta del usuario directamente, mejorando el recall semántico.    
    def generate_hypothetical_questions(self, text: str, n: int = 4) -> list:

        response = self.ai.chat(
            f"Generate exactly {n} questions that the following scientific text "
            f"would answer. Focus on: devices, signals, AI models, challenges "
            f"and recommendations. Return ONLY the questions, one per line:\n\n{text}"
        )
        # Separamos por líneas y quitamos las vacías
        questions = [q.strip() for q in response.split("\n") if q.strip()]
        return questions[:n]
    
    # Clasificación selectiva: detecta qué entidades existen en el fragmento antes de extraerlas.
    def classify_entities_present(self, text: str) -> list:
        result = self.ai.structured_output(
            prompt=(
                f"Read this scientific text and identify which types "
                f"of entities are mentioned:\n\n{text}"
            ),
            schema=EntityClassification,
            system_prompt=(
                "You are an IoT health monitoring expert. "
                "Your task is to identify which entity types are present "
                "in the text. Choose ONLY from this exact list: "
                "Device, Signal, Technology, Architecture, "
                "AIModel, BigData, Challenge, Strength, "
                "Recommendation, FutureWork. "
                "Do NOT invent new types. "
                "If none are present return an empty list."
            )
        )

        # Filtrar solo los tipos válidos del dominio
        valid_types = [
            t for t in result.entity_types
            if t in VALID_ENTITY_TYPES
        ]

        return valid_types
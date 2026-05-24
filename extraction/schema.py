from typing import List, Optional
from pydantic import BaseModel, Field

# ─────────────────────────────────────────
# ENTIDADES DEL DOMINIO IoT SALUD
# Cada clase representa un tipo de nodo en Neo4j
# ─────────────────────────────────────────

# Dispositivo físico que porta el paciente.
class Device(BaseModel):    
    name: str = Field(description="Name of the wearable device used in the paper")
    brand: Optional[str] = Field(default=None)
    type: Optional[str] = Field(default=None, description="Smartphone, Smartwatch, Bracelet, Ring")
    description: Optional[str] = Field(default=None,
        description="Describe how this device is used in the paper")

# Señal fisiológica capturada por el dispositivo.
class Signal(BaseModel):    
    name: str = Field(description="ECG, SpO2, Heart Rate")
    description: Optional[str] = Field(default=None,
        description="Describe how this signal is captured or used")
    
# Tecnología de comunicación o software empleada en el sistema IoT.
class Technology(BaseModel):
    name: str = Field(description="Name of the technology or protocol")
    tech_type: Optional[str] = Field(default=None,
        description="Wireless, Protocol, Software, Database, Platform")
    description: Optional[str] = Field(default=None,
        description="Describe how this technology is used in the paper")

# Arquitectura del sistema IoT propuesta en el artículo.
class Architecture(BaseModel):    
    type: str = Field(description="Fog, Cloud, Edge, Hybrid")
    platform: Optional[str] = Field(default=None, description="Azure, AWS, GCP")
    layers: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None,
        description="Describe how this architecture is proposed")

# Modelo de inteligencia artificial o machine learning.
class AIModel(BaseModel):
    name: str = Field(description="Name of the AI or ML model")
    model_type: Optional[str] = Field(default=None,
        description="Classification, Prediction, Clustering, Detection")
    description: Optional[str] = Field(default=None,
        description="Describe how this model is applied in the paper")

# Tecnología o técnica de Big Data mencionada en el artículo.
class BigData(BaseModel):
    name: str = Field(description="Name of the Big Data technique or processing method used in the paper")
    status: Optional[str] = Field(default=None, description="Implemented, Proposed")
    tool: Optional[str] = Field(default=None, description="Hadoop, Spark, Flink")
    description: Optional[str] = Field(default=None,
        description="Describe how Big Data is used in the paper")

# Limitaciones o desafíos identificados en el artículo.
class Challenge(BaseModel):
    description: str
    severity: Optional[str] = Field(default="Moderate", description="Critical, Moderate, Low")
    barrier_type: Optional[str] = Field(
        default="Technical",
        description="Technical, Ethical, Privacy, Interoperability"
    )

# Fortaleza o ventaja del sistema propuesto.
class Strength(BaseModel):
    description: str
    impact: Optional[str] = Field(default="Medium", description="High, Medium, Low")

# Recomendación técnica ofrecida por los autores.
class Recommendation(BaseModel):
    description: str
    focus: Optional[str] = Field(
        default="Algorithm",
        description="Hardware, Algorithm, Security, Protocol"
    )

# Línea de trabajo futuro propuesta en el artículo.
class FutureWork(BaseModel):
    description: str
    area: Optional[str] = Field(
        default="AI",
        description="AI, Sensors, Privacy, Protocols, Architecture"
    )

# Identifica qué temas aparecen en el texto. La IA revisa el artículo y hace una lista de la información 
# encontrada (como dispositivos o modelos de IA) para organizar la extracción.
class EntityClassification(BaseModel):
    entity_types: List[str] = Field(
        default_factory=list,
        description=(
            "List of entity types present in the text. "
            "Choose ONLY from: Device, Signal, Technology, "
            "Architecture, AIModel, BigData, Challenge, "
            "Strength, Recommendation, FutureWork"
        )
    )

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CONTENEDORES DE EXTRACCIÓN
# Definen qué información debe extraer el modelo en cada consulta. 
# Son los formularios de respuesta que le damos a la IA para que devuelva datos estructurados que podamos guardar en Neo4j. 
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Entidades de hardware físico e infraestructura IoT
class HardwareResult(BaseModel):
    devices: List[Device] = Field(default_factory=list)
    signals: List[Signal] = Field(default_factory=list)
    technologies: List[Technology] = Field(default_factory=list)
    architectures: List[Architecture] = Field(default_factory=list)

# Entidades de modelos analíticos y Big Data.
class AnalyticsResult(BaseModel):
    ai_models: List[AIModel] = Field(default_factory=list)
    bigdata: List[BigData] = Field(default_factory=list)

# Entidades de impacto cualitativo del estudio.
class ImpactResult(BaseModel):
    challenges: List[Challenge] = Field(default_factory=list)
    strengths: List[Strength] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    future_works: List[FutureWork] = Field(default_factory=list)
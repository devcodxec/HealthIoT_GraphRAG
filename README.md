# HealthIoT GraphRAG

Sistema Graph RAG para análisis sistemático de literatura científica sobre monitorización de salud con dispositivos IoT.

---

## Requisitos previos

Antes de instalar el proyecto asegúrate de tener instalado:

- Python 3.12+
- Neo4j Desktop o Neo4j Community Edition
- Ollama (para ejecutar llama3.2 en local)

---

## Paso 1 — Clonar el proyecto

```bash
git clone https://github.com/devcodxec/HealthIoT_GraphRAG
cd HealthIoT_GraphRAG
```

---

## Paso 2 — Crear el entorno virtual

```bash
python -m venv .venv
```

Activar el entorno:

```bash
# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## Paso 3 — Instalar las librerías

```bash
pip install -r requirements.txt
```

| Librería | Función |
|----------|---------|
| `streamlit` | Interfaz web del sistema |
| `neo4j` | Conexión con la base de datos Neo4j |
| `ollama` | Conexión con el modelo llama3.2 en local |
| `pymupdf` | Extracción de texto de los PDFs |
| `pydantic` | Validación de esquemas JSON del LLM |
| `python-dotenv` | Lectura de variables de entorno del .env |
| `tqdm` | Barra de progreso en el pipeline |

Si no tienes el archivo `requirements.txt` instala manualmente:

```bash
pip install streamlit neo4j ollama pymupdf pydantic python-dotenv tqdm
```

---

## Paso 4 — Configurar el archivo `.env`

Crea el archivo `.env` en la raíz del proyecto con estos valores:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=tu_usuario
NEO4J_PASSWORD=tu_contraseña
OLLAMA_MODEL=llama3.2
NEO4J_DATABASE=health-iot
```

## Paso 5 — Configurar Ollama

```bash
# Instalar llama3.2
ollama pull llama3.2

# Verificar que está disponible
ollama list
```


## Paso 6 — Configurar Neo4j

1. Abre **Neo4j Desktop**
2. Crea una nueva base de datos llamada `health-iot`
3. Inicia la base de datos
4. Verifica que el puerto `7687` está disponible

---

## Paso 7 — Añadir los PDFs

Copia los artículos científicos en formato PDF a la carpeta:

```
HealthIoT_GraphRAG/
└── data/
    └── scopus/
        ├── article01.pdf
        ├── article02.pdf
        └── ...
```
---

## Paso 8 — Ejecutar el sistema

### Opción A — Interfaz web (recomendada para demo)

```bash
streamlit run app.py --server.fileWatcherType none
```

Se abre automáticamente en `http://localhost:8501`

### Opción B — Menú terminal (para desarrollo)

```bash
python main.py pipeline     # construir el grafo desde PDFs
python main.py retrievers   # probar los 3 retrievers
python main.py agent        # probar el agente completo
python main.py eval         # evaluación estructurada completa
```

---

## Para qué sirve cada archivo principal

**`app.py`** es la interfaz web del sistema construida con Streamlit. Integra los 4 hitos del proyecto en 5 secciones accesibles desde el navegador sin necesidad de tocar el código.

**`main.py`** es el menú de terminal para desarrolladores. Permite ejecutar cada hito del sistema de forma independiente mediante argumentos de línea de comandos, útil para depuración y pruebas rápidas sin abrir el navegador.

---

## Estructura del proyecto


```
HealthIoT_GraphRAG/
├── app.py                  ← interfaz web Streamlit
├── main.py                 ← menú terminal
├── .env                    ← configuración (no subir a git
├── data/
│   └── scopus/             ← PDFs de los artículos
├── extraction/             ← chunking y extracción de entidades
├── graph/                  ← conexión y queries Neo4j
├── models/                 ← gestión del modelo llama3.2
├── retrievers/             ← los 4 retrievers
├── agents/                 ← los 4 agentes
├── evaluation/             ← benchmark de 15 preguntas
└── source/                 ← extracción de texto PDF
```






































































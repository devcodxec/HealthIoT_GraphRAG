import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Esta clase gestiona la conexión con la base de datos Neo4j.
# Expone run_query() para ejecutar cualquier consulta Cypher.
# Crea los índices y constraints necesarios.

class Neo4jManager:

    def __init__(self):
        uri      = os.getenv("NEO4J_URI")
        user     = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

    def close(self):
        self.driver.close()

    def test_connection(self):
        try:
            self.driver.verify_connectivity()
            print("Conexión exitosa con Neo4j")
        except Exception as e:
            print(f"Error de conexión: {e}")

    # Método genérico para ejecutar cualquier consulta Cypher con parámetros y devolver resultados como lista de diccionarios.
    def run_query(self, query: str, parameters: dict = None) -> list:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
        
    # Crea los constraints únicos y el índice vectorial.
    # Se llama UNA SOLA VEZ al inicio del proyecto para preparar Neo4j.
    # Constraints: evitan nodos duplicados con el mismo id/name.
    # Vector Index: permite búsqueda semántica por similitud de coseno sobre los embeddings de los Chunks (HyQE).
    def setup_schema(self):
        
        constraints = [
            # Cada artículo tiene un id único
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            # Los siguientes nodos se identifican por nombre
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Signal) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:AIModel) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BigData) REQUIRE b.name IS UNIQUE",
            # Los chunks tienen id propio
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.run_query(c)
            except Exception as e:
                print(f"  Constraint ya existe: {e}")

        # Índice vectorial para búsqueda semántica sobre los Chunks
        # dimensions=768 para nomic-embed-text, 1536 para text-embedding-3-small
        vector_index = """
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        try:
            self.run_query(vector_index)
            print("Schema de Neo4j configurado correctamente")
        except Exception as e:
            print(f"  Índice vectorial ya existe: {e}")
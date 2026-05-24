import os
import json
from typing import Type
from pydantic import BaseModel
import ollama
from dotenv import load_dotenv

load_dotenv()

# Clase que habla con el modelo de IA local (Ollama).
# Tiene dos modos:
#  - chat()            → respuesta libre en texto
#  - structured_output() → respuesta forzada a seguir un esquema Pydantic

class OllamaManager:
    

    def __init__(self):
        # Lee el modelo del .env, por defecto usa llama3
        self.model = os.getenv("OLLAMA_MODEL", "llama3")

    # Conversación simple:  pasamos un prompt y devuelve texto. Se puede usar para resumir entidades, relaciones, etc.
    def chat(self, prompt: str) -> str:

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error en chat: {e}")
            return ""


    # Pedimos al modelo que responda EXACTAMENTE con la estructura de un esquema Pydantic que le pasamos (Article, Device, etc.).
    # Ollama acepta format=schema.model_json_schema() para forzar JSON válido.
    # Después validamos con Pydantic para asegurar tipos correctos.
    def structured_output(self, prompt: str, schema: Type[BaseModel],
                          system_prompt: str = "") -> BaseModel:
     
        messages = []

        # Añadimos el system prompt si existe (describe el rol del modelo)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                format=schema.model_json_schema(),  # Ollama fuerza la estructura JSON
            )
            raw = response["message"]["content"]

            # Limpiamos posibles marcas de markdown que algunos modelos añaden
            clean = raw.replace("```json", "").replace("```", "").strip()

            # Pydantic valida y convierte el JSON a un objeto Python tipado
            return schema.model_validate_json(clean)

        except Exception as e:
            print(f"Error en structured_output: {e}")
            # Devolvemos instancia vacía para no romper el pipeline
            return schema.model_construct()
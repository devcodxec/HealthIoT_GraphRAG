import fitz  # PyMuPDF
import os
import re
from typing import Dict, Any

# Limpia el texto extraído del PDF eliminando:
# - Líneas con HTML o metadatos
# - Líneas muy cortas (ruido)
# - Espacios y saltos de línea excesivos
def clean_pdf_text(text: str) -> str:
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Eliminar líneas vacías o muy cortas
        if len(line) < 4:
            continue

        # Eliminar líneas con HTML o metadatos
        if any(tag in line for tag in [
            "text/html", "application/", "<!DOCTYPE",
            "<html", "<head", "<body", "<script",
            "ld+json", "<?xml", "http://", "https://"
        ]):
            continue

        # Eliminar líneas que son solo números o símbolos
        if re.match(r'^[\d\s\.\,\-\_\(\)]+$', line):
            continue

        clean_lines.append(line)

    # Unir líneas y eliminar espacios múltiples
    clean_text = "\n".join(clean_lines)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    return clean_text

# Lee un PDF y devuelve todo su texto como string.
# fitz (PyMuPDF) es el más robusto para artículos científicos.
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Limpiar el texto antes de devolverlo
        return clean_pdf_text(text)

    except Exception as e:
        print(f"  Error leyendo {pdf_path}: {e}")
        return ""

# Extrae metadatos básicos del nombre del archivo.
# Scopus suele nombrar los PDFs con el título o ID del artículo.
def get_metadata_from_filename(pdf_path: str) -> Dict[str, Any]:
  
    filename = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(filename)[0]

    return {
        "title":     name_without_ext,
        "source_db": "Scopus",
        "filename":  filename,
    }

# Carga todos los PDFs de una carpeta.
# Devuelve lista de diccionarios con texto y metadatos.
def load_pdfs_from_folder(folder_path: str) -> list:
    
    papers = []

    # Listamos todos los archivos .pdf de la carpeta
    pdf_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ])

    print(f"PDFs encontrados en {folder_path}: {len(pdf_files)}")

    for filename in pdf_files:
        pdf_path = os.path.join(folder_path, filename)

        print(f"  Leyendo: {filename}")
        text = extract_text_from_pdf(pdf_path)

        # Solo procesamos si el PDF tiene contenido
        if len(text.strip()) < 100:
            print(f"  Saltando {filename}: texto muy corto o vacío")
            continue

        metadata = get_metadata_from_filename(pdf_path)
        papers.append({
            "text":     text,
            "metadata": metadata
        })

    return papers
import requests

def clasificar_acoso(texto_descripcion: str) -> bool:
    """
    Usa un modelo Ollama local para clasificar si una interacción puede ser acoso.
    """
    prompt = f"""Dado este escenario: "{texto_descripcion}", responde únicamente con 'sí' o 'no'. ¿Hay indicios de acoso o comportamiento inapropiado?"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",  # Cambia por el modelo que estés usando
                "prompt": prompt,
                "stream": False
            },
            timeout=15
        )
        respuesta = response.json()["response"].strip().lower()
        return "sí" in respuesta or "si" in respuesta
    except Exception as e:
        print("Error con Ollama:", e)
        return False

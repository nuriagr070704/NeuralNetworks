import os
import random
import string
import zipfile

# Función para generar palabras con letras aleatorias al principio y al final
def generar_palabra_aleatoria(base, longitud_total=20):
    letras_restantes = longitud_total - len(base)
    letras_inicio = random.randint(0, letras_restantes)  # Cantidad aleatoria al inicio
    letras_final = letras_restantes - letras_inicio  # Resto al final
    inicio = ''.join(random.choices(string.ascii_letters, k=letras_inicio))
    final = ''.join(random.choices(string.ascii_letters, k=letras_final))
    return inicio + base + final

# Función para generar archivos en lotes
def crear_archivos_en_lotes(palabra, carpeta, inicio, fin, lote_tamano=1000):
    os.makedirs(carpeta, exist_ok=True)
    
    for i in range(inicio, fin, lote_tamano):
        for j in range(i, min(i + lote_tamano, fin)):
            contenido = generar_palabra_aleatoria(palabra, 20)
            file_path = os.path.join(carpeta, f"{j}.txt")  # Nombres como 10000.txt, ..., 49999.txt
            with open(file_path, "w") as file:
                file.write(contenido)

# Función para comprimir los archivos en un ZIP
def comprimir_archivos(carpeta, zip_name):
    with zipfile.ZipFile(zip_name, "w") as zipf:
        for file_name in os.listdir(carpeta):
            file_path = os.path.join(carpeta, file_name)
            zipf.write(file_path, arcname=file_name)

# Configuración inicial
palabras = ["piedra", "papel", "tijera"]
base_directorio = "archivos_generados"
os.makedirs(base_directorio, exist_ok=True)

# Generar y comprimir archivos para cada palabra
for palabra in palabras:
    carpeta = os.path.join(base_directorio, f"{palabra}_10000_to_49999")
    print(f"Generando archivos para '{palabra}' en: {carpeta}")
    crear_archivos_en_lotes(palabra, carpeta, 10000, 50000)
    zip_name = f"{palabra}_10000_to_49999.zip"
    print(f"Comprimiendo archivos en: {zip_name}")
    comprimir_archivos(carpeta, zip_name)
    print(f"Eliminando archivos temporales en: {carpeta}")
    for file_name in os.listdir(carpeta):
        os.remove(os.path.join(carpeta, file_name))
    os.rmdir(carpeta)

print("¡Proceso completado!")

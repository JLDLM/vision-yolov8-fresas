import os
import cv2
from ultralytics import YOLO

def cargar_modelo(ruta_modelo):
    """Carga el modelo entrenado de YOLOv8."""
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {ruta_modelo}")
    print(f"[INFO] Cargando modelo desde: {ruta_modelo}...")
    return YOLO(ruta_modelo)

def crear_video_desde_imagenes(ruta_carpeta_imagenes, modelo, ruta_video_salida, umbral_confianza=0.5, fps=2, segundos_por_foto=3):
    """
    Procesa todas las imágenes de una carpeta y genera un video estable,
    redimensionando los fotogramas para evitar cortes por tamaños diferentes.
    """
    extensiones_validas = (".jpg", ".jpeg", ".png", ".webp")
    
    if not os.path.exists(ruta_carpeta_imagenes):
        print(f"[ERROR] La carpeta '{ruta_carpeta_imagenes}' no existe.")
        return

    archivos = os.listdir(ruta_carpeta_imagenes)
    imagenes = sorted([f for f in archivos if f.lower().endswith(extensiones_validas)])

    if not imagenes:
        print(f"[WARN] No hay imágenes válidas en: {ruta_carpeta_imagenes}")
        return

    print(f"[INFO] Se encontraron {len(imagenes)} imágenes. Iniciando compilación estable...")

    video_writer = None
    repeticiones_por_frame = int(fps * segundos_por_foto)
    
    # Dimensiones estándar para asegurar compatibilidad total en el archivo de video
    ANCHO_ESTANDAR = 640
    ALTO_ESTANDAR = 640

    for nombre_imagen in imagenes:
        ruta_completa = os.path.join(ruta_carpeta_imagenes, nombre_imagen)
        
        # Ejecutar inferencia
        resultados = modelo(ruta_completa, conf=umbral_confianza, verbose=False)
        resultado = resultados[0]
        
        # .plot() genera la imagen con las máscaras/cajas
        frame_renderizado = resultado.plot()

        # REDIMENSIONAR FORZADO: Evita problemas si cambia el tamaño de la foto original
        frame_redimensionado = cv2.resize(frame_renderizado, (ANCHO_ESTANDAR, ALTO_ESTANDAR))

        # Inicializar el video_writer usando codec XVID
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(ruta_video_salida, fourcc, fps, (ANCHO_ESTANDAR, ALTO_ESTANDAR))
            print(f"[INFO] Creando archivo de video: {ruta_video_salida} ({ANCHO_ESTANDAR}x{ALTO_ESTANDAR}px)...")

        # Escribir el fotograma duplicado en el archivo de video (CORREGIDO SIN EL TYPO)
        for _ in range(repeticiones_por_frame):
            video_writer.write(frame_redimensionado)
            
        print(f"[PROCESADA EN VIDEO] -> {nombre_imagen}")

    if video_writer is not None:
        video_writer.release()
        print(f"\n[ÉXITO] Archivo de video guardado correctamente en: {ruta_video_salida}")
    else:
        print("[ERROR] No se pudo generar el archivo de video.")

if __name__ == "__main__":
    # CONFIGURACIÓN DE RUTAS LOCALES
    RUTA_MODELO = r"D:\Jair\Fresas6\runs\segment\train\weights\best.pt" 
    CARPETA_IMAGENES = r"D:\Jair\Fresas6\Imagenes" 
    VIDEO_SALIDA = r"D:\Jair\Fresas6\resultado_fresas.avi"
    
    try:
        modelo_yolo = cargar_modelo(RUTA_MODELO)
        
        # Cada imagen durará 2 segundos en el video para que rinda el set entero de 35 fotos
        crear_video_desde_imagenes(
            CARPETA_IMAGENES, 
            modelo_yolo, 
            VIDEO_SALIDA, 
            umbral_confianza=0.6, 
            fps=2, 
            segundos_por_foto=2
        )
        
    except Exception as e:
        print(f"[CRITICAL] Error en la ejecución: {e}")
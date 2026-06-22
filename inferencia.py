import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# modelo_path = r'C:\YOLOV8A25\runs\segment\uteroTodas200ep20\weights/best.pt'
# modelo_path = r'C:\YOLOV8A25\runs\segment\train\weights/best.pt'
modelo_path = r'C:\Users\CAPROM\Desktop\REYNA\runs\segment\uteromeromero2\weights/best.pt'
carpeta_entrada = r'C:\Users\CAPROM\Desktop\REYNA\Final\2.0ProcesoMiomas\ConMioma'
carpeta_salida = r'C:\Users\CAPROM\Desktop\REYNA\Final\2.0ProcesoMiomas\bbox\ConMioma'

os.makedirs(carpeta_salida, exist_ok=True)

modelo = YOLO(modelo_path)

extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp')

for ruta_img in Path(carpeta_entrada).glob('*'):
    if not ruta_img.suffix.lower() in extensiones_validas:
        continue

    nombre_archivo = ruta_img.stem
    imagen_original = cv2.imread(str(ruta_img))
    # imagen_con_contornos = imagen_original.copy()###

    resultados = modelo(imagen_original)[0]

    if resultados.masks is not None:
        for idx, mask in enumerate(resultados.masks.data):
            # Máscara y redimensionamiento
            mascara = mask.cpu().numpy().astype('float32')
            mascara = cv2.resize(mascara, (imagen_original.shape[1], imagen_original.shape[0]))
            mascara_binaria = (mascara > 0.5).astype('uint8') * 255

            # # Contornos
             contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)###
             cv2.drawContours(imagen_con_contornos, contornos, -1, (0, 255, 0), 2)###

            # #confianza
             x, y, w, h = cv2.boundingRect(mascara_binaria)###
             if resultados.boxes is not None and idx < len(resultados.boxes.conf):###
             confianza = resultados.boxes.conf[idx].item()###
                 texto_confianza = f"{confianza:.2f}"###
                 cv2.putText(imagen_con_contornos, texto_confianza, (x, y - 10),###
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)###

            # #POLÍGONO
             x, y, w, h = cv2.boundingRect(mascara_binaria)
             mascara_3c = cv2.merge([mascara_binaria]*3)
             recorte_poligono = cv2.bitwise_and(imagen_original, mascara_3c)
             recorte_poligono_crop = recorte_poligono[y:y+h, x:x+w]
             nombre_poligono = os.path.join(carpeta_salida, f"{nombre_archivo}_recorte_poligono_{idx}.jpg")
             cv2.imwrite(nombre_poligono, recorte_poligono_crop)

            #BBOX
            if resultados.boxes is not None and idx < len(resultados.boxes.xyxy):
                bbox = resultados.boxes.xyxy[idx].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                recorte_bbox = imagen_original[y1:y2, x1:x2]
                nombre_bbox = os.path.join(carpeta_salida, f"{nombre_archivo}_recorte_bbox_{idx}.jpg")
                cv2.imwrite(nombre_bbox, recorte_bbox)

    # # --- Guardar imagen con contornos y confianza ---
     salida_img = os.path.join(carpeta_salida, f"{nombre_archivo}_contornos.jpg")###
     cv2.imwrite(salida_img, imagen_con_contornos)###

print("✅ Solo recortes completados.")

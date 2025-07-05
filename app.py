import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown
from ultralytics import YOLO
from utils.distance_utils import calcular_distancia_real
from utils.ollama_utils import clasificar_acoso
from utils.alert_system import SistemaAlertas

# Configurar página
st.set_page_config(page_title="Detección Riesgos", layout="wide")

# Descargar modelos desde Google Drive si no existen
@st.cache_resource
def cargar():
    if not os.path.exists("modelos"):
        os.makedirs("modelos")

    # Descargar YOLOv8n
    yolo_id = "1T6MQ3jnTs-yPdLaeUYkCyV57uRTZywHM"
    yolo_path = "modelos/yolov8n.pt"
    if not os.path.exists(yolo_path):
        gdown.download(f"https://drive.google.com/uc?id={yolo_id}", yolo_path, quiet=False)

    # Descargar EfficientDet Lite
    tflite_id = "1U7TR-BvJSr0aRbWoTEMFaTvrvtT78CCa"
    tflite_path = "modelos/efficientdet_lite.tflite"
    if not os.path.exists(tflite_path):
        gdown.download(f"https://drive.google.com/uc?id={tflite_id}", tflite_path, quiet=False)

    # Cargar modelos
    yolo = YOLO(yolo_path)
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    return yolo, interp

# Iniciar webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Cargar modelos y sistema de alertas
yolo, tflite = cargar()
sistema = st.session_state.get('alertas', SistemaAlertas())
st.session_state['alertas'] = sistema

# Layout
col1, col2 = st.columns([3, 1])
vid_box = col1.empty()
stats = col2.container()
alertas = col2.container()

# Loop de video
frame_counter = 0
resultados_personas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # YOLO: seguimiento o detección
    if frame_counter % 3 == 0:
        try:
            resultados_personas = yolo.track(frame, persist=True)[0]
        except:
            resultados_personas = yolo(frame)[0]
    if resultados_personas is None:
        resultados_personas = yolo(frame)[0]

    # Personas detectadas
    cajas = []
    if resultados_personas.boxes is not None:
        for caja in resultados_personas.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, caja)
            cajas.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Armas con EfficientDet (filtrado por cajas humanas)
    inp_det = tflite.get_input_details()[0]
    out_boxes = tflite.get_output_details()[0]
    out_classes = tflite.get_output_details()[1]
    out_scores = tflite.get_output_details()[2]

    img = cv2.resize(frame, (320, 320))
    inp = np.expand_dims(img, axis=0).astype(np.uint8)
    tflite.set_tensor(inp_det['index'], inp)
    tflite.invoke()

    boxes_t = tflite.get_tensor(out_boxes['index'])[0]
    classes_t = tflite.get_tensor(out_classes['index'])[0]
    scores_t = tflite.get_tensor(out_scores['index'])[0]

    armas = []
    for i in range(len(scores_t)):
        if scores_t[i] > 0.6 and classes_t[i] == 0:
            y1, x1, y2, x2 = boxes_t[i]
            x1 = int(x1 * frame.shape[1])
            x2 = int(x2 * frame.shape[1])
            y1 = int(y1 * frame.shape[0])
            y2 = int(y2 * frame.shape[0])

            # Verificar si está dentro de una persona
            arma_en_persona = False
            for px1, py1, px2, py2 in cajas:
                if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                    arma_en_persona = True
                    break

            if not arma_en_persona:
                armas.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "ARMA", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Interacciones cercanas
    interacciones = []
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            d = calcular_distancia_real(cajas[i], cajas[j], frame.shape)
            if d < 1.5:
                interacciones.append((cajas[i], cajas[j], d))
                cx1 = (cajas[i][0] + cajas[i][2]) // 2
                cy1 = (cajas[i][1] + cajas[i][3]) // 2
                cx2 = (cajas[j][0] + cajas[j][2]) // 2
                cy2 = (cajas[j][1] + cajas[j][3]) // 2
                cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

    # Activar alertas
    if armas:
        sistema.activar("ARMA DETECTADA")
    elif interacciones and sistema.registrar(len(interacciones)):
        desc = f"{len(interacciones)} interacciones cercanas, distancia promedio {np.mean([d for *_, d in interacciones]):.2f} m"
        if clasificar_acoso(desc):
            sistema.activar("POSSIBLE ACOSO")

    # Mostrar frame
    vid_box.image(frame, channels="BGR", use_column_width=True)

    # Métricas
    with stats:
        st.metric("Personas Detectadas", len(cajas))
        st.metric("Interacciones Cercanas", len(interacciones))
        st.metric("Armas Detectadas", len(armas))

    # Estado de alerta
    with alertas:
        if sistema.alerta:
            st.warning(sistema.tipo)
        else:
            st.success("Situación Normal")

cap.release()

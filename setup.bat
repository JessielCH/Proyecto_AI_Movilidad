@echo off
REM Descargar modelos
if not exist modelos (
    mkdir modelos
)
curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -o modelos/yolov8n.pt
curl -L https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite -o modelos/efficientdet_lite.tflite
echo Modelos descargados correctamente.
pause

import cv2
import numpy as np

# Carregar YOLO
net = cv2.dnn.readNet("C:/Users/Cliente/Documents/TrabalhoCG/yolos/yolov3-spp.weights", "C:/Users/Cliente/Documents/TrabalhoCG/yolos/yolov3-spp.cfg")
classes = []

with open("C:/Users/Cliente/Documents/TrabalhoCG/yolos/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obter os nomes das camadas de saída
layer_names = net.getUnconnectedOutLayersNames()

# Inicializar o vídeo
video_path = "C:/Users/Cliente/Documents/TrabalhoCG/videos/Velozes e Furiosos - Desafio em Tokyo _ edit.mp4"
cap = cv2.VideoCapture(video_path)

# Inicializar o rastreador KCF
tracker = cv2.TrackerKCF_create()

# Inicializar o rastreador CSRT
csrt_tracker = cv2.TrackerCSRT_create()

# Modo inicial: detecção
mode = 'detection'
best_mechanism = 'detecção'
det_time = 0
track_time = 0
det_count = 0
track_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    start_time = cv2.getTickCount()

    if mode == 'detection':
        # Detectar objetos usando YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        # Informações sobre detecções
        class_ids = []
        confidences = []
        boxes = []

        # Processar saídas de YOLO
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Coordenadas do objeto detectado
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Coordenadas do retângulo
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Supressão não máxima para remover detecções duplicadas
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Desenhar caixas delimitadoras e rótulos no frame
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        det_time += (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        det_count += 1

    elif mode == 'tracking':
        # Atualizar o rastreador KCF com o novo frame
        if 'bbox' in locals():
            if bbox != (0, 0, 0, 0):  # Verificar se a ROI é válida
                success, bbox = tracker.update(frame)

                # Desenhar a bounding box no frame
                if success:
                    (x, y, w, h) = tuple(map(int, bbox))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("Seleção cancelada. ROI inválida.")

        track_time += (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        track_count += 1

    # Exibir o frame resultante
    cv2.putText(frame, f'Mecanismo Atual: {best_mechanism}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Detection and Tracking", frame)

    # Sair quando a tecla 'q' for pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        # Trocar para o modo de rastreamento
        mode = 'tracking'
        bbox = cv2.selectROI(frame, False)
        if bbox != (0, 0, 0, 0):  # Verificar se a ROI é válida
            tracker.init(frame, bbox)
            best_mechanism = 'rastreamento'
    elif key == ord('d'):
        # Trocar para o modo de detecção
        mode = 'detection'
        best_mechanism = 'detecção'

# Calcular média de tempo por frame
avg_det_time = det_time / det_count if det_count > 0 else 0
avg_track_time = track_time / track_count if track_count > 0 else 0

print(f'Média de tempo de detecção: {avg_det_time:.2f} segundos por frame')
print(f'Média de tempo de rastreamento: {avg_track_time:.2f} segundos por frame')

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

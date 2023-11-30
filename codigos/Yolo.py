
import cv2
import numpy as np

# Carregar YOLO
net = cv2.dnn.readNet("C:/Users/Cliente/Documents/TrabalhoCG/yolos/yolov3-spp.weights", "C:/Users/Cliente/Documents/TrabalhoCG/yolos/yolov3-spp.cfg")
classes = []

with open("C:/Users/Cliente/Documents/TrabalhoCG/yolos/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obter os nomes das camadas de saída
layer_names = net.getUnconnectedOutLayersNames()

# Carregar vídeo
cap = cv2.VideoCapture("C:/Users/Cliente\Documents/TrabalhoCG/videos/Velozes e Furiosos - Desafio em Tokyo _ edit.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detectar objetos usando YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Informações sobre detecções
    class_ids = []
    confidences = []
    boxes = []  # Adicionando a definição da lista 'boxes' aqui

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

    # Exibir o frame resultante
    cv2.imshow("YOLO Video", frame)

    # Sair quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
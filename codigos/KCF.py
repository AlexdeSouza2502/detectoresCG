import cv2

# Inicializar o rastreador KCF
tracker = cv2.TrackerKCF_create()

# Inicializar o vídeo
video_path = "C:/Users/Cliente/Documents/TrabalhoCG/videos/Brians Training .mp4"
cap = cv2.VideoCapture(video_path)

# Ler o primeiro frame
ret, frame = cap.read()
if not ret:
    print("Erro ao ler o vídeo")
    exit()

# Selecionar a região de interesse (ROI) para rastrear
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

while True:
    # Ler o próximo frame
    ret, frame = cap.read()
    if not ret:
        break

    # Atualizar o rastreador com o novo frame
    success, bbox = tracker.update(frame)

    # Desenhar a bounding box no frame
    if success:
        (x, y, w, h) = tuple(map(int, bbox))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibir o frame resultante
    cv2.imshow("Tracking", frame)

    # Sair quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

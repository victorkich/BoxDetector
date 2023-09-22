import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

# Abre três câmeras. Seus índices podem variar. Aqui estou assumindo 0, 1 e 2.
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(3)
cap3 = cv2.VideoCapture(4)

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)  # Verde
font_scale = 0.7
thickness = 2

while True:
    # Lê frames das três câmeras
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    success3, frame3 = cap3.read()

    # Verifica se todos os frames foram capturados com sucesso
    if success1 and success2 and success3:
        # Processa cada frame com o modelo YOLO
        results1 = model(frame1)
        results2 = model(frame2)
        results3 = model(frame3)

        # Visualiza os resultados em cada frame
        annotated_frame1 = results1[0].plot()
        annotated_frame2 = results2[0].plot()
        annotated_frame3 = results3[0].plot()

        # Combina os frames horizontalmente
        combined_frame = cv2.hconcat([annotated_frame1, annotated_frame2, annotated_frame3])

        # Mostra o frame combinado
        cv2.imshow("YOLOv8 Inference", combined_frame)

        # Encerra o loop se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera os objetos de captura e fecha todas as janelas
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()

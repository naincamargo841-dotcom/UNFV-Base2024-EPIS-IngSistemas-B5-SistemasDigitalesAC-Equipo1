import cv2
import mediapipe as mp
import pymsgbox

pymsgbox.alert("Presiona ESC para cerrar el programa")
camera = cv2.VideoCapture(0)

#Adjustando formato para mas fluidez
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    800
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    600
)

# NUEVO
# Crear acceso al módulo de detección facial

mp_face = mp.solutions.face_detection


# NUEVO
# Crear detector facial
# min_detection_confidence:confianza mínima para considerar como rostro


detector = mp_face.FaceDetection(

    min_detection_confidence = 0.5

)

while True:

    ret, frame = camera.read()

    if not ret:

        break


    # NUEVO
    # OpenCV trabaja BGR
    # MediaPipe trabaja RGB
    # Convertimos colores

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    # NUEVO
    # Enviar imagen al detector facial

    results = detector.process(

        rgb

    )


    # NUEVO
    # ¿Se detectaron rostros?

    if results.detections:


        # NUEVO
        # recorrer cada rostro detectado

        for detection in results.detections:


            # NUEVO
            # obtener caja delimitadora

            bbox = detection.location_data.relative_bounding_box


            # NUEVO
            # obtener tamaño de la imagen

            h, w, _ = frame.shape


            # NUEVO
            # Convertir posiciones del rostro
            # a pixeles

            x = int(

                bbox.xmin * w

            )

            y = int(

                bbox.ymin * h

            )

            width = int(

                bbox.width * w

            )

            height = int(

                bbox.height * h

            )


            # NUEVO
            # dibujar rectángulo del rostro

            cv2.rectangle(

                frame,

                (x,y),

                (x+width,y+height),

                (0,255,0),

                2

            )


    cv2.imshow(

        "Deteccion Facial",

        frame

    )

    if cv2.waitKey(30) & 0xFF == 27:

        break

camera.release()

cv2.destroyAllWindows()
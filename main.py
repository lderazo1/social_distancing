# ARCHIVO DE EJECUCION SISTEMA DISTANCIAMIENTO SOCIAL
# Autores: Jose Pullaguari - Luis Daniel Erazo.

# Uso métodos basado en proyecto MEDIUM-Detector distanciamiento social 
# https://sanju-mehla3599.medium.com/social-distancing-detector-using-opencv-and-deep-learning-ab712c1882fc

import numpy as np
import imutils
import time
import cv2
import os
import math
from itertools import chain 
from variables_globales import *

#Importe libreria
NOMBRES = open(YOLOV3_NOMBRES).read().strip().split('\n')
np.random.seed(123) #Semilla
COLORES = np.random.randint(0, 255, size=(len(NOMBRES), 3), dtype='uint8')
print('LIBRERIA YOLO V3')#Conexion exitosa

#Carga modelo YOLO, requiere peso,configuraciones y nombres.
red_neuronal = cv2.dnn.readNetFromDarknet(YOLOV3_CONFIGURACIONES, YOLOV3_PESOS)#metodo lectura modelo

#nombres de salida que componen a Yolo
nombres_detectados = red_neuronal.getLayerNames()
nombres_detectados = [nombres_detectados[i[0] - 1] for i in red_neuronal.getUnconnectedOutLayers()]

#flujo entrada de video
video_capturado = cv2.VideoCapture(VIDEO_PRUEBA) #Metodo de OpenCV
msj = None
(b, a) = (None, None)

#Configuracion dimensiones cuadro de video
try:
    if(imutils.is_cv2()):
        video_frame = cv2.cv.CV_CAP_PROP_FRAME_COUNT #Contar numero de cuadros por toma
    else:
        video_frame = cv2.CAP_PROP_FRAME_COUNT
    total = int(video_capturado.get(video_frame))
    print('Cuadros por toma detectados: ', total)
except Exception as e:
    print(e)
    total = -1

#Lectura de cada frame del video de entrada
while True:
    (base_frame, frame) = video_capturado.read()

    if not base_frame:
        break
    
    if b is None or a is None:
        a, b = (frame.shape[0], frame.shape[1])

    #Objeto binario
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False) #intercambiar las matrices de colores de R&B
    red_neuronal.setInput(blob)

    tiempo_inicial = time.time()
    salida_red = red_neuronal.forward(nombres_detectados)
    tiempo_final = time.time()
    
    marcos_objeto = []
    reservado = [] #Obligatoria para procesamiento.
    id_clase = []
    lineas_frame = []
    marcos_centrales = []

    #Repaso detecciones de objetos encontrados
    for output in salida_red:
        for detection in output:
            #deteccion hace referencia a similitud con clases coco
            similitud = detection[5:]
            clase_detectada = np.argmax(similitud)
            reserva = similitud[clase_detectada]
            #umbral de confianza 0.5
            if reserva > 0.5 and clase_detectada == 0:
                box = detection[0:4] * np.array([b, a, b, a])
                (coordenada_x_central, coordenada_y_central, ancho, altura) = box.astype('int')
                
                x = int(coordenada_x_central - (ancho / 2))
                y = int(coordenada_y_central - (altura / 2))
                
                marcos_centrales = [coordenada_x_central, coordenada_y_central]

                marcos_objeto.append([x, y, int(ancho), int(altura)])
                reservado.append(float(reserva))
                id_clase.append(clase_detectada)

    id_dectetado = cv2.dnn.NMSBoxes(marcos_objeto, reservado, 0.5, 0.3)
    #busqueda de frames delimitados
    if len(id_dectetado) > 0:
        no_cumple = []
        n = 0
        
        for i in id_dectetado.flatten():
            
            (x, y) = (marcos_objeto[i][0], marcos_objeto[i][1])
            (lg,at) = (marcos_objeto[i][2], marcos_objeto[i][3])
            coord_x = marcos_objeto[i][0] + (marcos_objeto[i][2] // 2)
            coord_y = marcos_objeto[i][1] + (marcos_objeto[i][3] // 2)

            color = [int(c) for c in COLORES[id_clase[i]]]
            tipo = '{}: {:.4f}'.format(NOMBRES[id_clase[i]], reservado[i])

            id_dectetado_copy = list(id_dectetado.flatten())
            id_dectetado_copy.remove(i)

            for j in np.array(id_dectetado_copy):
                coord_x2 = marcos_objeto[j][0] + (marcos_objeto[j][2] // 2)
                coord_y2 = marcos_objeto[j][1] + (marcos_objeto[j][3] // 2)
                #Distancia Euclidiana
                #Entre el centroide frame actual
                distancia = math.sqrt(math.pow(coord_x2 - coord_x, 2) + math.pow(coord_y2 - coord_y, 2))
                #Contra los demas centroides del frame detectado.
                if distancia <= DISTANCIA:
                    cv2.line(frame, (marcos_objeto[i][0] + (marcos_objeto[i][2] // 2), marcos_objeto[i][1]  + (marcos_objeto[i][3] // 2)), (marcos_objeto[j][0] + (marcos_objeto[j][2] // 2), marcos_objeto[j][1] + (marcos_objeto[j][3] // 2)), (0, 0, 255), 2)
                    no_cumple.append([coord_x2, coord_y2])
                    no_cumple.append([coord_x, coord_y])

            if coord_x in chain(*no_cumple) and coord_y in chain(*no_cumple):
                n += 1
                #rectangulos con las dimensiones del frame modelo
                cv2.rectangle(frame, (x, y), (x + lg, y + at), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + lg, y + at), (0, 255, 0), 2)

            cv2.putText(frame, tipo, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (50, 50), (450, 90), (0, 0, 0), -1)
            cv2.putText(frame, 'SIN DISTANCIAMIENTO: {}'.format(n), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)            


    if msj is None:

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        msj = cv2.VideoWriter(SALIDA, fourcc, 30,(frame.shape[1], frame.shape[0]), True)

        if total > 0:
            print('---------')
            elap = (tiempo_final - tiempo_inicial)
            print('Un frame tarda {:.4f} segundos en procesarse'.format(elap))
            print('Finalizaría en {:.4f} segundos aproximadamente'.format(elap * total))

    msj.write(frame)

print('--------')
print('Ejecutado con éxito')
msj.release()
video_capturado.release()                                
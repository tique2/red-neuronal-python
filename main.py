import numpy as np

class RedNeuronal:
    def __init__(self):
        # Inicialización de pesos y sesgos
        self.peso = np.random.rand(1)
        self.sesgo = np.random.rand(1)

    def predecir(self, x):
        return self.peso * x + self.sesgo

    def entrenar(self, datos_entrada, datos_salida, epocas, tasa_aprendizaje):
        for _ in range(epocas):
            for i in range(len(datos_entrada)):
                entrada = datos_entrada[i]
                objetivo = datos_salida[i]

                # Propagación hacia adelante
                prediccion = self.predecir(entrada)

                # Cálculo del error
                error = objetivo - prediccion

                # Actualización de pesos y sesgos
                self.peso += tasa_aprendizaje * error * entrada
                self.sesgo += tasa_aprendizaje * error


# Datos de entrada y salida
datos_entrada = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
datos_salida = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10,11])

# Hiperparámetros
epocas = 1000
tasa_aprendizaje = 0.01

# Crear y entrenar la red neuronal
red_neuronal = RedNeuronal()
red_neuronal.entrenar(datos_entrada, datos_salida, epocas, tasa_aprendizaje)

# Predicción para el próximo número en la secuencia
siguiente_numero = 19
prediccion = red_neuronal.predecir(siguiente_numero)
print("La predicción para el siguiente número en la secuencia es:", prediccion)
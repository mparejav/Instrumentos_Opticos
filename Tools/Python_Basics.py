# Este archivo contiene definiciones basicas de la programación orientada a objetos.
# Ejemplos de estructuras, clases, objetos, atributos, métodos, etc.
# Además de un sheet code con lo elemental de python.

####################### TIPOS DE VARIABLES #########################

# NÚMEROS
entero = 42                   # int: Números enteros, tamaño dinámico, precisión ilimitada
decimal = 3.14159             # float: Números decimales, precisión doble (64 bits)
complejo = 3 + 4j             # complex: Números complejos (parte real + imaginaria)

# TEXTO
cadena = "Hola mundo"         # str: Cadenas de texto, inmutables, codificación UTF-8
caracter = 'a'                # str: Un solo carácter también es string en Python

# BOOLEANOS
verdadero = True              # bool: Valores lógicos True/False, subclase de int
falso = False                 # False equivale a 0, True equivale a 1

# COLECCIONES
lista = [1, 2, 3, "texto"]    # list: Ordenada, mutable, permite duplicados
tupla = (1, 2, 3)             # tuple: Ordenada, inmutable, permite duplicados
conjunto = {1, 2, 3}          # set: No ordenado, mutable, sin duplicados
diccionario = {"clave": "valor"}  # dict: Pares clave-valor, mutable, claves únicas

# ESPECIALES
nulo = None                   # NoneType: Representa ausencia de valor
rango = range(5)              # range: Secuencia inmutable de números (0 a 4)

# VERIFICACIÓN DE TIPOS
print(f"Tipo de entero: {type(entero)}")        # type() devuelve el tipo
print(f"Es lista?: {isinstance(lista, list)}")   # isinstance() verifica tipo

####################### NUMPY ARRAYS #########################

import numpy as np  # Importación estándar de NumPy

# CREACIÓN DE ARRAYS
array_1d = np.array([1, 2, 3, 4])              # np.array(lista) - 1D: Array unidimensional desde lista
array_2d = np.array([[1, 2], [3, 4]])          # np.array(lista_2d) - 2D: Array bidimensional (matriz)
zeros = np.zeros(5)                             # np.zeros(tamaño) - Array de ceros: [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))                          # np.ones((filas, columnas)) - Array de unos: matriz 2x3
rango_np = np.arange(0, 10, 2)                  # np.arange(inicio, fin, paso) - Rango: [0 2 4 6 8]
espaciado = np.linspace(0, 1, 5)                # np.linspace(inicio, fin, cantidad) - 5 números entre 0 y 1

# PROPIEDADES DE ARRAYS
print(f"Forma: {array_2d.shape}")               # shape: dimensiones (2, 2)
print(f"Tamaño: {array_2d.size}")               # size: total de elementos (4)
print(f"Dimensiones: {array_2d.ndim}")          # ndim: número de dimensiones (2)
print(f"Tipo de datos: {array_2d.dtype}")       # dtype: tipo de datos (int64)

# INDEXING Y SLICING
elemento = array_1d[0]                          # array[índice] - Primer elemento: 1
fila = array_2d[0]                              # array[fila] - Primera fila: [1 2]
columna = array_2d[:, 1]                        # array[:, columna] - Segunda columna: [2 4]
segmento = array_1d[1:3]                        # array[inicio:fin] - Elementos de indice 1 a 2: [2 3]

# OPERACIONES MATEMÁTICAS
suma = array_1d + 5                             # Suma escalar: [6 7 8 9]
producto = array_1d * 2                         # Producto escalar: [2 4 6 8]
suma_arrays = array_1d + array_1d               # Suma elemento a elemento
producto_punto = np.dot(array_1d, array_1d)     # Producto punto: 30

# FUNCIONES ESTADÍSTICAS
promedio = np.mean(array_1d)                    # Media aritmética: 2.5
maximo = np.max(array_1d)                       # Valor máximo: 4
minimo = np.min(array_1d)                       # Valor mínimo: 1
suma_total = np.sum(array_1d)                   # Suma total: 10
desviacion = np.std(array_1d)                   # Desviación estándar

# RESHAPE Y MANIPULACIÓN
reshapeado = array_1d.reshape(2, 2)             # array.reshape(filas, columnas) - Cambiar forma: 2x2 matrix
aplanado = array_2d.flatten()                   # array.flatten() - Aplanar a 1D: [1 2 3 4]
transpuesto = array_2d.T                        # array.T - Transponer matriz
transpuesto2 = np.transpose(array_2d)           # np.transpose(array) - Transponer matriz (alternativo)

# FUNCIONES LÓGICAS
condicion = array_1d > 2                        # array > valor - Array booleano: [False False True True]
filtrado = array_1d[array_1d > 2]               # array[condición] - Filtrar: [3 4]
donde = np.where(array_1d > 2)                  # np.where(condición) - Índices donde es True: (array([2, 3]),)

####################### POO #########################

class Michi(): # Definición de la clase 
    especie = 'Felino' # Atributo de clase. No requiere instancia ni definición de objeto.

    def __init__(self, nombre, edad): # Método constructor. Se ejecuta al crear una instancia de la clase (self).

        self.nombre = nombre    # Atributo de instancia. Requiere instancia y definición de objeto.
        self.edad = edad

    def maullar(self):          # Método de instancia. Requiere, por lo menos, instancia y definición de objeto.
        print(f"{self.nombre} dice: ¡Miau!")
        
    def caminar(self, pasos):
        print(f"{self.nombre} ha caminado {pasos} pasos.")

mi_michi = Michi("Ramona", 1) # Definición de objeto (instancia de la clase Michi).

print(f"Mi michi se llama {mi_michi.nombre}, tiene {mi_michi.edad} años y es un {mi_michi.especie}.")

mi_michi.maullar() # Llamada al método maullar del objeto mi_michi.

mi_michi.caminar(5) # Llamada al método caminar del objeto mi_michi con argumento 5.
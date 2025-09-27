"""
We will plot the patterns of functions Jinc and Sinc, because these functions will be our reference to 
compare with results obtained in the codes of Fresnel Transform and Angular spectrum
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from Masks import *

# Parámetros de muestreo
Δ = 3.45  # um. Tamaño del pixel
N = 1024  # Número de muestras
L = N * Δ  # um. Tamaño físico de la grilla

# Crear coordenadas centradas en 0
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Parámetros del círculo
radius = 100  # um. Radio del círculo
λ = 0.6328  # um (longitud de onda He-Ne)
# Generar máscara circular
mask = circle(radius, X, Y)

# Crear coordenadas radiales
R = np.sqrt(X**2 + Y**2)

# Definir función Jinc
def Jinc(r, radius):
    """
    Calcula la función Jinc(r) = 2*J1(2*pi*r*radius/λ)/(2*pi*r*radius/λ)
    donde J1 es la función de Bessel de primera especie de orden 1
    
    Parameters:
        r: coordenada radial
        radius: radio de la apertura circular
    """

    k = 2*np.pi/λ  # número de onda
    
    # Argumento de la función de Bessel
    arg = k*radius*r/R.max()
    
    print (λ)
    # Evitar división por cero en r = 0
    return np.where(r == 0, 1.0, 2*special.j1(arg)/(arg))



# Calcular patrón de difracción teórico
I_teorico = np.abs(Jinc(R, radius))**2

# Graficar resultados
plot_fields(mask, I_teorico, x, y, x, y, Cut_Factor=30, title0="Máscara Circular", titlez="Patrón de Difracción Teórico")


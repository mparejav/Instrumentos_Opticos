import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   

#Dummy code to try to generate the difraction pattern using The Fresnel transformation method

"""
1. Take or generate the input field as U [n,m,0]
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the parabolic phases term
3. Calculate U'' [n,m,z] with FFT 
4. Calculate U [n,m,z] adding the spherical phase output terms
5. Re organize U [n,m,z] using shift
"""

#Inicializing important variables
λ = 0 #um.  Wavelength of light
L = 0 #um. This is the length of the grid that we use for the transmitance
N = 0 #Number of samples that we take 
Δ = 0 # um. Sampling interval in the spatial domain


"""1. Take or generate the input field as U [n,m,0]
"""
#Creating an optic field, in this case the field is a circular slit
radius = 1000 #um. The radius of the circle

#Giving a value to some variables
L = 2048 #Is useful that this value could be a 2 power
N = 1024 #Number of samples
λ = 0.5 #um. Wavelenght of light

#Creating the coordinates of the space

x = np.linspace (-L/2, L/2, N, endpoint =False)
y = np.linspace (-L/2, L/2, N, endpoint=False)






import numpy as np  
from Miscelanea import *
from Difraction_Implementation_Of_Matrix import *


# Light parameters
λ = 0.633  # um. Wavelength of light (He-Ne laser)

# Lens parameters
f = 500000 #um. Focal length of the lens 

#Sensor parameters CAM1 
Δ_1_CAM1 = 3.8 # um. The size of the pixel in the sensor
M_x_CAM1 = 4640 # Number of pixels in the x axis for the sensor
M_y_CAM1 = 3506 # Number of pixels in the y axis for the sensor


#Input sampling field parameters
Δ_fx = 1 / (M_x_CAM1 * λ) #um^-1. sampling interval in the frequences domain in axis fx
Δ_fy = 1 / (M_y_CAM1 * λ) #um^-1. sampling interval in the frequences domain in axis fy
L_0x = 1 / Δ_fx # um. Physical size of the input field grid in axis x
L_0y = 1 / Δ_fy  # um. Physical size of the input field grid in axis y

N = 1024 # Number of samples per side of the  grid 
Δ_0 = (λ*f)/(Δ_1_CAM1*N)  # um. Sampling interval in the spatial domain.


z_min = (N * Δ_0**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented    

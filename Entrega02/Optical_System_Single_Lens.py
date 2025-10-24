
import numpy as np  
from Miscelanea import *
from Difraction_Implementation_Of_Matrix import *

# Light parameters
lamda = 0.633  # um. Wavelength of light (He-Ne laser)

# Lens parameters
f = 500000 #um. Focal length of the lens 

#Distance between object and lens
d = 1000000 #um.

#Sensor parameters CAM2

delta_1_CAM2 = 5.2 # um. The size of the pixel in the sensor
M_x_CAM2 = 1280 # Number of pixels in the x axis for the sensor
M_y_CAM2 = 1024 # Number of pixels in the y axis for the sensor

#Input sampling field parameters
delta_fx = 1 / (M_x_CAM2 * lamda) #um^-1. sampling interval in the frequences domain in axis fx
delta_fy = 1 / (M_y_CAM2 * lamda) #um^-1. sampling interval in the frequences domain in axis fy
L_0x = 1 / delta_fx # um. Physical size of the input field grid in axis x
L_0y = 1 / delta_fy  # um. Physical size of the input field grid in axis y

N = 1024 # Number of samples per side of the  grid 
Δ_0 = (lamda*f)/(delta_1_CAM2*N)  # um. Sampling interval in the spatial domain.


z_min = (N * Δ_0**2) / lamda #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented   

#Creating the parameteres for the matrix system
A,B,C, D = transferMatrix_Propagation_Lens_Propagation(f,d)

#Creating the coordinates for the input field
x_0 = (np.arange(M_x_CAM2)-M_x_CAM2/2) * Δ_0
y_0 = (np.arange(M_y_CAM2)-M_y_CAM2/2) * Δ_0
X_0,Y_0 = np.meshgrid (x_0,y_0)

#Creating the input field
U_0 = circle(5000,X_0,Y_0)

#Calculating the output field with the diffractive formulation
U_B, x_1, y_1 = difractive_formulation (U_0,A,B,D,lamda,Δ_0,delta_1_CAM2,X_0,Y_0,M_x_CAM2,M_y_CAM2)

"""
5. Calculate I[n,m,z] - the intensity at distance z
""" 
I_z = np.abs(U_B)**2  # Intensity is the magnitude squared of the field
I_z = I_z / I_z.max()  # Normalizing to its max value

I_0 = np.abs(U_0)**2  # Intensity at z = 0   

"""
Plotting the results
"""

plot_fields(I_0, I_z, x_0, y_0, x_1, y_1, Cut_Factor=40, title0 = "Transmitancia", titlez = "Intensidad del Campo propagado:\n formulación difractiva")

 


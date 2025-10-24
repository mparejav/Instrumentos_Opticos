import numpy as np  
from Miscelanea import *
from Difraction_Implementation_Of_Matrix import *

"""
Parameteres of the optical system with variable transmitance
"""
# Light parameters
λ = 0.633  # um. Wavelength of light (He-Ne laser)

# Lens parameters
f = 500000 #um. Focal length of the lens 

#Sensor parameters CAM1 
Δ_1_CAM1 = 3.8 # um. The size of the pixel in the sensor
M_x_CAM1 = 4640 # Number of pixels in the x axis for the sensor
M_y_CAM1 = 3506 # Number of pixels in the y axis for the sensor


"""
Parameters of the input field and the optical system
"""
#Input sampling field parameters
Δ_fx = 1 / (M_x_CAM1 * λ) #um^-1. sampling interval in the frequences domain in axis fx
Δ_fy = 1 / (M_y_CAM1 * λ) #um^-1. sampling interval in the frequences domain in axis fy
L_0x = 1 / Δ_fx # um. Physical size of the input field grid in axis x
L_0y = 1 / Δ_fy  # um. Physical size of the input field grid in axis y

#Parameters for the mirror M1
L_xM1 = 10400 # um. Size of the mirror in the x axis
L_yM1 = 5800  # um. Size of the mirror in the y axis

N = 1024 # Number of samples per side of the  grid 
Δ_0 = (λ*f)/(Δ_1_CAM1*N)  # um. Sampling interval in the spatial domain.


z_min = (N * Δ_0**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented    

"""
Parameteres for the output field at the propagation towards Transmitance M1
"""


"""
Creating the parameteres for the matrix system, in the first propagation towards The transmitance
"""

A, B, C,D = transferMatrix_Propagation_Lens_Propagation(f,f)

"""
Creating the input field and the output field when it is propagated to the transmitance M1
"""
#Creating the coordinates for the input field
x_0 = (np.arange(M_x_CAM1)-M_x_CAM1/2) * Δ_0
y_0 = (np.arange(M_y_CAM1)-M_y_CAM1/2) * Δ_0
X_0,Y_0 = np.meshgrid (x_0,y_0)

#Creating the input field, at this time it is a circular aperture
U_0 = circle(50000,X_0,Y_0)



#Calculating the output field with the diffractive formulation
U_beforeTransmitance, x_1, y_1 = difractive_formulation (U_0,A,B,D,λ,Δ_0,Δ_1_CAM1,X_0,Y_0,M_x_CAM1,M_y_CAM1)

"""
Creating the transmitance function and applying it to the output field U_beforeTransmitance
"""
#Creating the coordinates for the transmitance
x_M1 = np.linspace(-L_xM1/2, L_xM1/2, M_x_CAM1)
y_M1 = np.linspace(-L_yM1/2, L_yM1/2, M_y_CAM1)
X_M1, Y_M1 = np.meshgrid (x_M1,y_M1)
Δx_M1 = L_xM1 / M_x_CAM1
Δy_M1 = L_yM1 / M_y_CAM1

#Creating the transmitance function
Transmitance_M1 = transmitance_1 (L_xM1, L_yM1, X_M1, Y_M1)

#Applying the transmitance to the output field, using the propierty of convolution in the fourier domain
U_beforeTransmitancefft = (np.fft.fftshift (np.fft.fft2 (U_beforeTransmitance)))*(Δ_1_CAM1**2) #Fourier transform of the field before the transmitance
Transmitance_M1fft = (np.fft.fftshift (np.fft.fft2 (Transmitance_M1)))*(Δ_1_CAM1**2)  #Fourier transform of the transmitance
U_after_M1fft = U_beforeTransmitancefft * Transmitance_M1fft  #Applying the transmitance in the fourier domain
U_after_M1 = (np.fft.fftshift (np.fft.ifft2 (U_after_M1fft))) *Δ_fx*Δ_fy #Field after the transmitance in the spatial domain

#Creating the parameteres for the matrix system, in the second propagation towards the sensor CAM1
A2, B2, C2,D2 = transferMatrix_Propagation_Lens_Propagation(f,f)

#Calculating the output field with the diffractive formulation
U_CAM1, x_CAM1, y_CAM1 = difractive_formulation (U_beforeTransmitance,A2,B2,D2,λ,Δ_0,Δ_1_CAM1,X_0,Y_0,M_x_CAM1,M_y_CAM1)

"""
Calculating the intensities
"""
#Intensity of the input field
I_0 = np.abs(U_0)**2
#Intensity at the sensor CAM1
I_CAM1 = np.abs(U_after_M1)**2
#Normalization of the intensity
I_CAM1 = I_CAM1 / np.max(I_CAM1)

"""
Plotting the results
"""

plot_fields(I_0, I_CAM1, x_0, y_0, x_1, y_1, Cut_Factor=40, title0 = "Objeto", titlez = "Intensidad del Campo propagado:\n formulación difractiva")




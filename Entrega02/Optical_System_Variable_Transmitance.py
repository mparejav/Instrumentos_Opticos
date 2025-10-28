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

"""
1. Parameters of the input field and the optical system
"""

N = 1024 # Number of samples per side of the  grid
L_0 = 50000  # um. Physical size of the input field grid
Δ_0 = L_0 / N  # um. Sampling interval in the spatial domain

#Creating the coordinates for the input field
x_0, y_0, X_0, Y_0 = coordinates (L_0, L_0, N,N)

"""
2. Parameters for the propagation towards Transmitance M1
"""
Δ_f1 = 1 / L_0 #um^-1. sampling interval in the frequences domain
M_1 = 1 / (λ * Δ_f1) # Number of samples to represent the signal per axis
Δ_1 = (λ *f)/(N*Δ_0)  # um. Sampling interval in the output field           
L_1 = N * Δ_1 #um. Physical size of the grid in the output field    

"""
Important terms to get a good sammpling
"""
z_min = (N * Δ_0**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented

#Creating the coordinates for the first propagation
x_1, y_1, X_1, Y_1 = coordinates (L_1, L_1, N,N)

"""
3. The paremeters for the last propagation towards the sensor CAM1
"""
Δ_f2 = 1 / L_0 #um^-1. sampling interval in the frequences domain
M_2 = 1 / (λ * Δ_f2) # Number of samples to represent the signal per axis
Δ_2 = (λ *f)/(N*Δ_1)  # um. Sampling interval in the output field           
L_2 = N * Δ_2 #um. Physical size of the grid in the output field    

"""
Important terms to get a good sammpling
"""
z_min = (N * Δ_1**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_1)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented

#Creating the coordinates for the first propagation
x_2, y_2, X_2, Y_2 = coordinates (L_2, L_2, N,N)

"""
4. This parameters will be used for the last coordinates at the sensor CAM1.
This is the field stop of the first system
"""
#Sensor parameters CAM1 
Δ_1_CAM1 = 3.8 # um. The size of the pixel in the sensor
M_x_CAM1 = 4640 # Number of pixels in the x axis for the sensor
M_y_CAM1 = 3506 # Number of pixels in the y axis for the sensor
L_xCAM1 = M_x_CAM1 * Δ_1_CAM1  # um. Physical size of the sensor CAM1 in the x axis
L_yCAM1 = M_y_CAM1 * Δ_1_CAM1  # um. Physical size of the sensor CAM1 in the y axis

#Creating the coordinates for the sensor CAM1
x_CAM1, y_CAM1, X_CAM1, Y_CAM1 = coordinates (L_xCAM1, L_yCAM1, M_x_CAM1,M_y_CAM1)

"""
5. Parameteres of the Transmitance M1. It is the aperture stop of the first system
"""
L_xM1 = 10400 # um. Physical size of the transmitance M1 in the x axis
L_yM1 = 5800 # um. Physical size of the transmitance M1 in the y axis

#Creating the coordinates for the transmitance M1
x_M1, y_M1, X_M1, Y_M1 = coordinates (L_xM1, L_yM1, N,N)

"""
6. Parameters for the CAM2. This is the field and aperture stop of the second system
"""

Δ_1_CAM2 = 5.2 # um. The size of the pixel in the sensor
M_x_CAM2 = 1280 # Number of pixels in the x axis for the sensor
M_y_CAM2 = 1024 # Number of pixels in the y axis for the sensor
L_xCAM2 = Δ_1_CAM2 * 1280 #um. Physical size of the sensor CAM2 in the x axis
L_yCAM2 = Δ_1_CAM2 * 1024 #um. Physical size of the sensor CAM2 in the x axis

#Creating the coordinates for this sensor
x_CAM2, y_CAM2, X_CAM2, Y_CAM2 = coordinates (L_xCAM2, L_yCAM2, M_x_CAM2, M_y_CAM2)


"""
Creating the parameteres for the matrix system, in the first propagation towards The transmitance
In the second propagation towards the sensor CAM1, the parameters ABCD will be same as here
"""

A, B, C,D = transferMatrix_Propagation_Lens_Propagation(f,f)

"""
Creating the input field and the output field when it is propagated to the transmitance M1
"""

#Creating the input field, at this time it is a circular aperture
U_0 = circle(1000,X_0,Y_0) #With first coordinates

#Taking an image as the input field
U_0 = load_image(r'Entrega02\Noise _images\Noise (1).png', N)

#Calculating the output field with the diffractive formulation
#Here we have the spectrum of U_0
U_beforeTransmitance = difractive_formulation (U_0,A,B,D,λ,Δ_0,X_0,Y_0,X_1,Y_1)

"""
Creating the transmitance function and applying it to the output field U_beforeTransmitance
"""

#Creating the transmitance function
Transmitance_M1 = transmitance_1 (L_xM1,L_yM1, X_M1, Y_M1)

#Multiplying the field before the transmitance by the transmitance function
U_afterTransmitance = U_beforeTransmitance * Transmitance_M1


#Calculating the output field with the diffractive formulation
U_CAM1 = difractive_formulation (U_afterTransmitance,A,B,D,λ,Δ_1,X_1,Y_1,X_2, Y_2)

#Organizing the output field at the sensor CAM1 with shifting
U_CAM1 = np.fft.fftshift(U_CAM1)


"""
Calculating the intensities
"""
#Intensity of the input field
I_0 = np.abs(U_0)**2
#Intensity at the sensor CAM1
I_CAM1 = np.abs(U_CAM1)**2

#Normalization of the intensity
if (np.max(I_CAM1) ==0):
    I_CAM1 = I_CAM1
else:
    I_CAM1 = I_CAM1 / np.max(I_CAM1)

#Intensity at the sensor CAM2, when we take the square module, we lose the terms of the spherical
#phase terms, then the intensity of the propagated field is equal when d is different to f, if d = f

I_CAM2 = np.abs(U_beforeTransmitance)**2
#Normalization of the intensity
I_CAM2 = I_CAM2 / np.max(I_CAM2)
 # logaritmic scale
I_CAM2 = np.log10(I_CAM2 + 1e-12) 

"""
Plotting the results
"""
#We plot the intensity of the propagated field towards CAM2
plot_fields(I_0, I_CAM2, x_0, y_0, x_CAM2, y_CAM2, Cut_Factor=40, title0 = "Objeto", titlez = "Intensidad del Campo propagado \n en CAM2")

#We plot the intensity of the input field and the intensity at the sensor CAM1 with the coordinates of the CAM1
plot_fields(I_0, I_CAM1, x_0, y_0, x_CAM1, y_CAM1, Cut_Factor=40, title0 = "Objeto", titlez = "Intensidad del Campo propagado\n en CAM1")




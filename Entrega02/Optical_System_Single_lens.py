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

#Distance between the object and Lens L2
d = 7000 #um.

"""
1. Parameters for the CAM2. This is the field and aperture stop of the second system
"""

Δ_1_CAM2 = 5.2 # um. The size of the pixel in the sensor
M_x_CAM2 = 1280 # Number of pixels in the x axis for the sensor
M_y_CAM2 = 1024 # Number of pixels in the y axis for the sensor
L_xCAM2 = Δ_1_CAM2 * 1280 #um. Physical size of the sensor CAM2 in the x axis
L_yCAM2 = Δ_1_CAM2 * 1024 #um. Physical size of the sensor CAM2 in the x axis

#Creating the coordinates for this sensor
x_CAM2, y_CAM2, X_CAM2, Y_CAM2 = coordinates (L_xCAM2, L_yCAM2, M_x_CAM2, M_y_CAM2)


"""
2. Parameters of the propagated field towards the transmitance or DMD
"""
Δ_0x = (λ *(d+f))/(M_x_CAM2*Δ_1_CAM2)  # um. Sampling interval in the output field in x axis
Δ_0y = (λ *(d+f))/(M_y_CAM2*Δ_1_CAM2)  # um. Sampling interval in the output field in y axis         
L_0x = M_x_CAM2 * Δ_0x #um. Physical size of the grid in the output field in x axis      
L_0y = M_y_CAM2 * Δ_0y #um. Physical size of the grid in the output field in y axis

#Creating the coordinates for the first propagation
x_0, y_0, X_0, Y_0 = coordinates (L_0x, L_0y, M_x_CAM2,M_y_CAM2)


"""
Creating the parameteres for the matrix system, in the first propagation towards The transmitance
In the second propagation towards the sensor CAM1, the parameters ABCD will be same as here
"""

A, B, C,D = transferMatrix_Propagation_Lens_Propagation(f,d)

"""
Creating the input field and the output field when it is propagated to the transmitance M1
"""

#Creating the input field, at this time it is a circular aperture
U_0 = circle(1000,X_0,Y_0) #With first coordinates

#Taking an image as the input field
U_0 = load_image(r'Entrega02\Noise _images\Noise (1).png', M_x_CAM2,M_y_CAM2)

#Calculating the output field with the diffractive formulation
#Here we have the spectrum of U_0
U_CAM2 = difractive_formulation (U_0,A,B,D,λ,Δ_0x,Δ_0y,X_0,Y_0,X_CAM2,Y_CAM2)


"""
Calculating the intensities
"""
#Intensity of the input field
I_0 = np.abs(U_0)**2
#Intensity at the sensor CAM1
I_CAM2 = np.abs(U_CAM2)**2

#Normalization of the intensity
#We need that the max value of I_CAM1 would be differente of 0
if (np.max(I_CAM2) ==0):
    I_CAM2 = I_CAM2
else:
    I_CAM2 = I_CAM2 / np.max(I_CAM2)

I_CAM2 = np.log10(I_CAM2 + 1e-12) 

"""
Plotting the results
"""
#We plot the intensity of the propagated field towards CAM2
plot_fields(I_0, I_CAM2, x_0, y_0, x_CAM2, y_CAM2, Cut_Factor=40, title0 = "Objeto", titlez = "Intensidad del Campo propagado \n en CAM2")





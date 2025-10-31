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
1. Parameters of the CAM1
We will define the coordinates of the propagated fields with the parameters of the CAM1 
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
2. Parameters of the propagated field towards the transmitance or DMD
"""
Δ_2x = (λ *2*f)/(M_x_CAM1*Δ_1_CAM1)  # um. Sampling interval in the output field in x axis
Δ_2y = (λ *2*f)/(M_y_CAM1*Δ_1_CAM1)  # um. Sampling interval in the output field in y axis         
L_2x = M_x_CAM1 * Δ_2x #um. Physical size of the grid in the output field in x axis      
L_2y = M_y_CAM1 * Δ_2y #um. Physical size of the grid in the output field in y axis

#Creating the coordinates for the first propagation
x_2, y_2, X_2, Y_2 = coordinates (L_2x, L_2y, M_x_CAM1,M_y_CAM1)

"""
3. Parameters of the DMD
"""
Δ_M1 = 5.4 #um. Sampling interval of the sensor CAM1
N_M1x = 1080 #Number of micromirrors in axis x
N_M1y = 1920 #Number of micromirrors in axis y
L_xM1 = 10400 # um. Physical size of the transmitance M1 in the x axis
L_yM1 = 5800 # um. Physical size of the transmitance M1 in the y axis


"""
Creating the parameteres for the matrix system, in the first propagation towards The transmitance
In the second propagation towards the sensor CAM1, the parameters ABCD will be same as here
"""

A, B, C,D = transferMatrix_Propagation_Lens_Propagation(f,f)

"""
Creating the input field and the output field when it is propagated to the transmitance M1
"""

#Creating the input field, at this time it is a circular aperture
U_0 = circle(1000,X_CAM1,Y_CAM1) #With first coordinates

#Taking an image as the input field
U_0 = load_image(r'Entrega02\Noise _images\Noise (6).png', M_x_CAM1,M_y_CAM1)

#Calculating the output field with the diffractive formulation
#Here we have the spectrum of U_0
U_beforeTransmitance_M1 = difractive_formulation (U_0,A,B,D,λ,Δ_1_CAM1,Δ_1_CAM1,X_CAM1,Y_CAM1,X_2,Y_2)

"""
Creating the transmitance function and applying it to the output field U_beforeTransmitance
"""

#Creating the transmitance function
Transmitance_M1 = transmitance_1 (L_xM1,L_yM1,910,1190, X_2, Y_2)

#We need that the U_beforeTransmitance and Transmitance_M1 have the same number of samples


#Multiplying the field before the transmitance by the transmitance function
U_afterTransmitance_M1 = U_beforeTransmitance_M1 * Transmitance_M1


#Calculating the output field with the diffractive formulation
U_CAM1 = difractive_formulation (U_afterTransmitance_M1,A,B,D,λ,Δ_2x,Δ_2y,X_2,Y_2,X_CAM1, Y_CAM1)

#Organizing the output field at the sensor CAM1 with shifting
U_CAM1 = np.fft.fftshift(U_CAM1)


"""
Calculating the intensities
"""
"""
This part is just if we want to plot the fields before and after the transmitance DMD
"""
#Intensity of the field before transmitance
I_beforeTransmitance_M1 = np.abs(U_beforeTransmitance_M1)**2
if (np.max(I_beforeTransmitance_M1) ==0):
    I_beforeTransmitance_M1 = I_beforeTransmitance_M1
else:
    I_beforeTransmitance_M1 = I_beforeTransmitance_M1 / np.max(I_beforeTransmitance_M1)
    
I_beforeTransmitance_M1 = np.log10(I_beforeTransmitance_M1 + 1e-12) 

#Intensity at the field after Transmitance
I_afterTransmitance_M1 = np.abs(U_afterTransmitance_M1)**2

#Normalization of the intensity
#We need that the max value of I_afterTransmitance_M1 would be differente of 0
if (np.max(I_afterTransmitance_M1) ==0):
    I_afterTransmitance_M1 = I_afterTransmitance_M1
else:
    I_afterTransmitance_M1 = I_afterTransmitance_M1 / np.max(I_afterTransmitance_M1)

I_afterTransmitance_M1 = np.log10(I_afterTransmitance_M1 + 1e-12) 

"""
At this part we calculate for the intensities of the field in the CAM1
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


"""
Plotting the results
"""

#We plot the intensity of the field before and after transmitance
#plot_fields(I_beforeTransmitance_M1, I_afterTransmitance_M1, x_2, y_2, x_2, y_2, Cut_Factor=40, title0 = "Intensidad de campo antes\n de M1", titlez = "Intensidad del Campo después \n de M1")

#We plot the intensity of the input field and the intensity at the sensor CAM1 with the coordinates of the CAM1
plot_fields(I_0, I_CAM1, x_CAM1, y_CAM1, x_CAM1, y_CAM1, Cut_Factor=60, title0 = "Objeto", titlez = "Intensidad del Campo propagado\n en CAM1")




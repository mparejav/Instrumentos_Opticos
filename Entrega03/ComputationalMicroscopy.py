import numpy as np  
from Miscelanea import *
from Difraction_Implementation_Of_Matrix import *

"""
Parameteres of the optical system with variable pupile
"""
# Light parameters
λ = 0.533  # um. Wavelength of light (He-Ne laser)

# Objective parameters
fMO = 500000 #um. Focal length of the lens 

#Tube lens parameteres
fTL = 200000 #um. Focal distance of TL

#Propagation distance for the first difractive implementation
L_0 = 2*fMO

#Propagation distance for the second difractive implementation
L_1 = 2*fTL


"""
1. Parameters of the CAM1
We will define the coordinates of the propagated fields with the parameters of the CAM1 
"""
#Sensor parameters CAM1 
Δ_1_CAM1 = 2.74 # um. The size of the pixel in the sensor
M_CAM1 = 2848 # Number of pixels in the x and y axis for the sensor
L_CAM1 = M_CAM1 * Δ_1_CAM1  # um. Physical size of the sensor CAM1 in the x and y axis

#Creating the coordinates for the sensor CAM1
x_CAM1, y_CAM1, X_CAM1, Y_CAM1 = coordinates (L_CAM1, L_CAM1, M_CAM1,M_CAM1)

"""
2. Parameters of the propagated field towards the pupile
"""
Δ_2  = (λ *2*L_1)/(M_CAM1*Δ_1_CAM1)  # um. Sampling interval in the output field in x and y axis        
L_2 = M_CAM1 * Δ_2 #um. Physical size of the grid in the output field in x axis      

#Creating the coordinates for the first propagation
x_2, y_2, X_2, Y_2 = coordinates (L_2, L_2, M_CAM1,M_CAM1)

"""
3. Parameters of the input Field
"""
Δ_3  = (λ *2*L_0)/(M_CAM1*Δ_2)  # um. Sampling interval in the output field in x and y axis        
L_3 = M_CAM1 * Δ_3 #um. Physical size of the grid in the output field in x axis 
#Creating the coordinates for the first propagation
x_3, y_3, X_3, Y_3 = coordinates (L_3, L_3, M_CAM1,M_CAM1)

"""
Creating the parameteres for the matrix system, in the first propagation towards The transmitance
In the second propagation towards the sensor CAM1, the parameters ABCD will be same as here
"""

A1, B1, C1,D1 = transferMatrix_Propagation_Lens_Propagation(fMO,fMO)

"""
Creating the input field and the output field when it is propagated to the transmitance M1
"""

#Creating the input field, at this time it is a circular aperture
U_0 = circle(1000,X_CAM1,Y_CAM1) #With first coordinates

#Taking an image as the input field
U_0 = load_image(r'Entrega02\Noise _images\Noise (6).png', M_CAM1,M_CAM1)


#Calculating the output field with the diffractive formulation
#Here we have the spectrum of U_0
U_beforepupile = difractive_formulation (L_0,U_0,A1,B1,D1,λ,Δ_1_CAM1,Δ_1_CAM1,X_CAM1,Y_CAM1,X_2,Y_2)

"""
Creating the transmitance function and applying it to the output field U_beforeTransmitance
"""

#Creating the transmitance function
pupile = transmitance_ring (L_3,L_3,0,1100, X_2, Y_2)
#pupile = transmitance_1 (L_xM1,L_yM1,X_2,Y_2)
#pupile = transmitance_X_rect(X_2, Y_2, 150, 900, L_xM1, L_yM1)

#We need that the U_beforeTransmitance and pupile have the same number of samples


#Multiplying the field before the transmitance by the transmitance function
U_afterpupile = U_beforepupile * pupile

A2, B2, C2,D2 = transferMatrix_Propagation_Lens_Propagation(fTL,fTL)

#Calculating the output field with the diffractive formulation
U_CAM1 = difractive_formulation (L_1,U_afterpupile,A2,B2,D2,λ,Δ_2,Δ_2,X_2,Y_2,X_CAM1, Y_CAM1)

#Organizing the output field at the sensor CAM1 with shifting
U_CAM1 = np.fft.fftshift(U_CAM1)


"""
Calculating the intensities
"""
"""
This part is just if we want to plot the fields before and after the transmitance DMD
"""
#Intensity of the field before transmitance
I_beforepupile = np.abs(U_beforepupile)**2
if (np.max(I_beforepupile) ==0):
    I_beforepupile = I_beforepupile
else:
    I_beforepupile = I_beforepupile / np.max(I_beforepupile)
    
I_beforepupile = np.log10(I_beforepupile + 1e-7) 

#Intensity at the field after Transmitance
I_afterpupile = np.abs(U_afterpupile)**2

#Normalization of the intensity
#We need that the max value of I_afterpupile would be differente of 0
if (np.max(I_afterpupile) ==0):
    I_afterpupile = I_afterpupile
else:
    I_afterpupile = I_afterpupile / np.max(I_afterpupile)

I_afterpupile = np.log10(I_afterpupile + 1e-12) 

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
#plot_fields(I_beforepupile, I_afterpupile, x_2, y_2, x_2, y_2, Cut_Factor=40, title0 = "Intensidad de campo antes\n de M1", titlez = "Intensidad del Campo después \n de M1")

#We plot the intensity of the input field and the intensity at the sensor CAM1 with the coordinates of the CAM1
plot_fields(I_afterpupile, I_CAM1, x_2, y_2, x_CAM1, y_CAM1, Cut_Factor=60, title0 = "Espectro filtrado por la \n transmitancia", titlez = "Intensidad del Campo propagado\n en CAM1")




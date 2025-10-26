import numpy as np  
from Miscelanea import *


#In this module we will use the Fresnel transformation method with the parameters of ABCD matrix
 

def difractive_formulation (input_field,A,B,D,λ,initial_delta,X_0,Y_0,X_1, Y_1):
    k = 2 * np.pi / λ  # um^-1. Wavenumber
    #Firstly we create the spherical terms for the input field
    SphericalInput = np.exp(((1j*k*A)/(2*B))*((X_0)**2+ (Y_0)**2))
    #We multiply the input field per the spherical input term
    U_1 = input_field * SphericalInput
    
    #We are calculating the fft for U_1
    U_2 = np.fft.fftshift((np.fft.fft2(U_1))* (initial_delta**2))
    
    #We create the spherical phase output term
    SphericalOutput = (np.exp((1j*k*D/(2*B))*(((X_1)**2)+ ((Y_1)**2))))/(1j*λ*B)
    
    #We need that SphericalOutput has the same shape as U_2, then we will pad the U_2 with zeros
    U_2_padded = pad (U_2, SphericalOutput)
 
    #We multiply the U_2 per sphericalOutput term
    U_B = SphericalOutput * U_2_padded
    
    return U_B








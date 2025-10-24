import numpy as np  
from Miscelanea import *


#In this module we will use the Fresnel transformation method with the parameters of ABCD matrix
 

def difractive_formulation (U_0,A,B,D,λ,Δ_0,Δ_1,X_0,Y_0,M_x,M_y):
    k = 2 * np.pi / λ  # um^-1. Wavenumber
    #Firstly we create the spherical terms for the input field
    SphericalInput = np.exp(((1j*k*A)/(2*B))*((X_0)**2+ (Y_0)**2))
    #We multiply the input field per the spherical input term
    U_1 = U_0 * SphericalInput
    #We are calculating the fft for U_1
    U_2 = np.fft.fftshift(np.fft.fft2(U_1)) * (Δ_0**2)
    
    
    #We create the coordinates for our propagated field
    x_1 = (np.arange(M_x) - M_x/2) * Δ_1
    y_1 = (np.arange(M_y) - M_y/2) * Δ_1
    X_1,Y_1 = np.meshgrid (x_1,y_1)
    X_1, Y_1 = np.meshgrid(x_1, y_1)
    
    #We create the spherical phase output term
    SphericalOutput = (np.exp((1j*k*D/(2*B))*(((X_1)**2)+ ((Y_1)**2))))/(1j*λ*B)
    #We multiply the U_2 per sphericalOutput term
    U_B = SphericalOutput * U_2
    
    return U_B, x_1, y_1








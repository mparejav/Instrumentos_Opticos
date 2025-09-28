import numpy as np 
import matplotlib.pyplot as plt  
from Masks import *
from scipy import special
from scipy.special import fresnel
from scipy.integrate import quad_vec

"""
Functions for analytical solutions
"""

"""
In this part we are creating the anaytical solution for the circular aperture using the Jinc function
"""

#Function for generate the analytical solution of the circular aperture
def U_of_r(r,radius, z, λ,k):
    # we are integrating from 0 to the radius of the aperture
    def integrand(rho, r):
        return rho * np.exp(1j * k * rho**2 / (2*z)) * special.j0(k * r * rho / z)

    #We are using quad_vec to integrate the function
    val, _ = quad_vec(lambda rho: integrand(rho, r), 0, radius, limit=200)

    prefac = (2*np.pi * np.exp(1j*k*z)) / (1j * λ * z) * np.exp(1j * k * r**2 / (2*z))
    return prefac * val

"""
In this part we are creating the analytical solution for the square aperture using the Sinc function
The input field is composed by the product of two rect functions, one in x and other in y
The Fresnel transform is linear, so we can calculate the Fresnel transform of each rect function 
and then multiply them to get the final result
"""
def I_axis(coord, a, z, k):
    # Computing the argument for the Fresnel integrals
    sqrt_term = np.sqrt(k/(2*z))
    # Calculating u+ and u-
    u_plus  = np.sqrt(2/np.pi) * (sqrt_term*a + sqrt_term*coord)
    u_minus = np.sqrt(2/np.pi) * (-sqrt_term*a + sqrt_term*coord)

    # Fresnel en u+ y u-
    C_plus, S_plus = fresnel(u_plus)
    C_minus, S_minus = fresnel(u_minus)
    
    # Calculating the differences
    deltaC = C_plus - C_minus
    deltaS = S_plus - S_minus

    # Prefactor
    prefac = np.sqrt(np.pi*z/k) * np.exp(-1j*k*coord**2/(2*z))
    return prefac * (deltaC + 1j*deltaS)

"""
We will calculate the correlation between the numerical and analytical solutions
We will use the Teorem of Convolution - correlation in the Fourier space
"""
def calculate_correlation(A, B):
    Fourier_Uz = np.fft.fft2(A)
    Fourier_U_sinc = np.fft.fft2(B)
    #Map of correlation
    correlation = np.fft.fftshift(np.fft.ifft2(Fourier_Uz * np.conj(Fourier_U_sinc)))
   
    # Normalization factor
    norm_factor = np.sqrt(np.sum(np.abs(A)**2) * np.sum(np.abs(B)**2))

    # Maximum correlation value normalized
    C = np.max(np.abs(correlation)) / norm_factor

    return C * 100  # Return as percentage

#Holaa
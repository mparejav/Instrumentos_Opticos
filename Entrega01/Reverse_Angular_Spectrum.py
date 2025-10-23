import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   
from Miscelanea import *
"""
This code determines the spatial structure of an unknown transmitance and the distance to the sensor in a reverse process
using the Angular Spectrum Method.
"""

# Light parameters
λ = 0.6328  # um. Wavelength of light (He-Ne laser)
k = 2 * np.pi / λ  # um^-1. Wavenumber

# Sensor parameters 
Δ = 5 # um. Sampling interval in the spatial domain. 

# Setup parameters
z = 87500 # um. Propagation distance (Transmitance to Sensor)
z_mm = z / 1000 # mm
print("Propagation distance set at:", z_mm, "mm")

#M = 1/(λ*Δf) # Number of samples to represent the signal per axis 

# Conditions to asure proper sampling
#f_max = M*Δf  # um^-1. Maximum spatial frequency
f_Nyquist = 1/(2*Δ)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented

"""
1.) First, we need the propagated field at distance z. However, the sensor only measures intensity, which is the square of the absolute value of the field.
This fact makes not posible to recover the original input field, since the phase information is lost. 
We can recover some information about the input field, but not all of it. This probably will affect the quality of the result.  
"""
# ------- Distance_Source-Transmitance ---- Distance_Transmitance-Sensor
# Path 1:        38 mm                           77 mm
# Path 2:        38 mm                           111 mm
# Path 3:        38 mm                           156 mm
# Path 4:        46 mm                           109 mm
# Path 5:        152 mm                          109 mm

path = select_image_path(option = 4)

I_z = Intensity_Field_Diffraction_Pattern(path)

U_z = np.sqrt(I_z) # We assume this is the field at distance z. Phase term is cero.

N_y, N_x = I_z.shape  # Number of pixels along y and x-axis of the image
L_x = N_x * Δ  # um. Physical size of the sensor grid along x-axis
L_y = N_y * Δ  # um. Physical size of the sensor grid along y-axis

# Sampling parameters
Δf_x = 1 / (Δ*N_x)  # um^-1. Sampling interval in the frequency domain. Spectral discretization 
Δf_y = 1 / (Δ*N_y)  # um^-1.

z_max = (N_y * Δ**2) / λ # um. Maximum propagation distance in which angular spectrum method is well sampled. We'll take the smaller N to be safe

if(z > z_max):
    # It will be just an alert, nothing will change 
    print("Exceded maximum propagation distance for proper sampling in the angular spectrum method.")

"""
2.) Calculate A[p,q,z] - the angular spectrum at distance z using FFT 
""" 

A_z = Δ**2 * np.fft.fft2(U_z)  # um^2. Angular spectrum at distance z

"""
3.) Divide by the Transfer Function to get A[p,q,0] - the angular spectrum at distance 0
"""

"""
Due to the fact that we are creating the meshgrid for the frequencies using the np.fft.fftfreq function 
theres no need to manually shift the zero frequency component.
"""
# Arrays of spatial frequencies 
fx = np.fft.fftfreq(N_x, d = Δ)   # frequencies along x ;     d : sample spacing in the ORIGINAL DOMAIN (Δx)
fy = np.fft.fftfreq(N_y, d = Δ)   # frequencies along y

# 2D grids for fx and fy
FX, FY = np.meshgrid(fx, fy)

# Argument of the square root in the Transfer Function
arg = 1 - (λ*FX)**2 - (λ*FY)**2

# Mask: Only compute where arg >= 0 to avoid evanescent waves and problems with sqrt of negative numbers
mask = arg >= 0

# Initialize Transfer_Function with zeros
Transfer_Function = np.zeros_like(FX, dtype = np.complex128)

Transfer_Function[mask] = np.exp(-1j * z * k * np.sqrt(arg[mask]))

# Calculate A[p,q,0] using the "inverse" Transfer Function
A_0 = A_z * Transfer_Function # Element-wise product

"""
4.) Calculate U[n,m,0] - the output field at distance 0 using inverse FFT
"""

U_0 = (Δf_x * Δf_y) * np.fft.ifft2(A_0)  # IFFT to obtain the input field at distance 0

U_0 = np.abs(U_0)  

"""
5.) Graph the results
"""

# Create physical coordinates centered at 0
x = np.linspace(-L_x/2, L_x/2, N_x, endpoint = False)
y = np.linspace(-L_y/2, L_y/2, N_y, endpoint = False)
X, Y = np.meshgrid(x, y)

Graph_Mask_and_Field_Angular_Spectrum(U_0, I_z, x, y, contrast_limit = 1, title_input = 'Recovered Transmitance', title_output = 'Captured Intensity field')
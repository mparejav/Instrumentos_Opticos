import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   
from Masks import *

# Dummy (ya no es dummy, pero quedará así) code to generate the diffraction pattern using the Angular Spectrum Method of a circular aperture
# Will try to emulate the structure given in class

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
-> 2.) Calculate A[p,q,0] - the angular spectrum at z=0 using FFT
-> 3.) Calculate A[p,q,z] - the angular spectrum at distance z using the Transfer Function
-> 4.) Re order the frequencies to center the zero frequency component
-> 5.) Calculate U[n,m,z] - the output field at distance z using inverse FFT
-> 6.) Calculate I[n,m,z] - the intensity at distance z
"""

"""
These values are defined by the initial conditions. 
Resolution and number of samples are set initially, but we can also defined the lenght of the grid
and stablish the other parameters in consequence.
"""
# Light parameters
λ = 0.6328  # um. Wavelength of light (He-Ne laser)
k = 2 * np.pi / λ  # um^-1. Wavenumber

# Sensor parameters (CS165MU1 Cmos sensor taken as reference)
Δ = 3.45 # um. Sampling interval in the spatial domain. (Square pixel size)
N = 1440 # Number of samples per side of the square grid 
L = N * Δ  # um. Physical size of the sensor grid (Emm...) ~ 5 mm 
print("Physical size of the grid L:", L, "um")

# Setup parameters
z = 20000 # um. Propagation distance 
z_mm = z / 1000 # mm
print("Propagation distance set at:", z_mm, "mm")

# Sampling parameters
Δf = 1 / (Δ*N)  # um^-1. Sampling interval in the frequency domain. Spectral discretization 
M = 1/(λ*Δf) # Number of samples to represent the signal per axis 

# Conditions to asure proper sampling
f_max = M*Δf  # um^-1. Maximum spatial frequency
z_max = (N * Δ**2) / λ # um. Maximum propagation distance in which angular spectrum method is well sampled
f_Nyquist = 1/(2*Δ)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented

if(z > z_max):
    print("Exceded maximum propagation distance for proper sampling in the angular spectrum method.")
    Propagation_Distance = z_max
    print("Propagation distance set at:", Propagation_Distance)

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
"""
# Create physical coordinates centered at 0
x = np.linspace(-L/2, L/2, N, endpoint = False)
y = np.linspace(-L/2, L/2, N, endpoint = False)
X, Y = np.meshgrid(x, y)

U_0 = circle(50, X, Y)
#U_0 = rectangle(60, 60, X, Y)
#U_0 = vertical_slit(40, X, Y)
#U_0 = horizontal_slit(40, X, Y)
#U_0 = cross_mask(80,80,60,40,60,20,N,X,Y)
#U_0 = load_image('Images/Rochi_square.png', N)

"""
-> 2.) Calculate A[p,q,0] - the angular spectrum at z=0 using FFT
"""

"""
Parameters in fft2:
----------------------
a : array_like
    Input array, can be complex.
    
s : sequence of ints, optional
- `s` sets the number of points that the FFT will have along each axis.
- If `s` is smaller than the input size: the input is cropped, reducing the array length.
- If `s` is larger than the input size: the input is zero-padded, increasing the array length.
- If `s` is None: the FFT uses the original input shape (no cropping or padding).

Conceptual link:
- Zero-padding in the spatial domain (larger `s`) leads to finer sampling in the frequency domain,
  i.e. a smaller frequency spacing Δf = 1/L.
- Cropping (smaller `s`) discards part of the input and broadens the frequency spacing.
- The choice of `s` therefore directly controls the spectral resolution and should be consistent
  with the physical grid definition (L = N·Δx).
"""

"""
Firstly, we try to add a padding factor to avoid aliasing effects due to circular convolution, however
we notice that using the np functions for FFT and similars the padding was not necessary. The results 
with a padding factor of 1 (no padding) were good enough.
"""
padding_factor = 1          # Increase this factor to increase padding
N_padded = N * padding_factor  # New size for FFT with padding. Digital padding.

# FFT2 computes the bidimensional Fast Fourier Transform
A_0 = (Δ**2) * np.fft.fft2(U_0, s=(N_padded, N_padded))  # second parameter of fft2 is the size of the FFT (Related with padding, look comment above)

"""
# -> 3.) Calculate A[p,q,z] - the angular spectrum at distance z using the Transfer Function
"""

A_z = np.zeros((N_padded, N_padded), dtype = np.complex128) # Initialize A[p,q,z]. Angular spectrum at distance z

"""
Due to the fact that we are creating the meshgrid for the frequencies using the np.fft.fftfreq function 
theres no need to manually shift the zero frequency component.
"""
# Arrays of spatial frequencies 
fx = np.fft.fftfreq(N_padded, d = Δ)   # frequencies along x ;     d : sample spacing in the ORIGINAL DOMAIN (Δx)
fy = np.fft.fftfreq(N_padded, d = Δ)   # frequencies along y

"""
np.fft.fftfreq:
---------------
- Generates the discrete frequency bins associated with an FFT of length n.
- Input:
    n : number of samples (FFT size, must match the array length or padded length)
    d : sample spacing in the ORIGINAL DOMAIN (Δx)
- Output:
    Array of frequencies (cycles per unit of d) of length n.
    For even n:
        f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)
    For odd n:
        f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (n*d)
- Note:
    These are the physical frequency values (fx, fy, ...) corresponding 
    to each FFT output index. They must be consistent with the FFT size 
    (including any padding) and the sampling interval.
"""

# 2D grids for fx and fy
FX, FY = np.meshgrid(fx, fy)

# Argument of the square root in the Transfer Function
arg = 1 - (λ*FX)**2 - (λ*FY)**2

# Mask: Only compute where arg >= 0 to avoid evanescent waves and problems with sqrt of negative numbers
mask = arg >= 0

# Initialize Transfer_Function with zeros
Transfer_Function = np.zeros_like(FX, dtype = np.complex128)

Transfer_Function[mask] = np.exp(1j * z * k * np.sqrt(arg[mask]))

# Calculate A[p,q,z] using the Transfer Function
A_z = A_0 * Transfer_Function # Element-wise product

"""
-> 4.) Calculate U[n,m,z] - the output field at distance z using inverse FFT
"""

U_z = (Δf**2) * np.fft.ifft2(A_z)  # Inverse FFT to get the output field at distance z

"""
-> 5.) Calculate I[n,m,z] - the intensity at distance z
""" 

I_z = np.abs(U_z)**2  

"""
Graph results   
"""

# espectro = np.fft.fftshift(np.log(np.abs(A_z)+1)) --> Proving something to analyze aliasing

Graph_Mask_and_Field_Angular_Spectrum(U_0, I_z, x, y, contrast_limit = 0.8, title_input = "Transmitance", title_output = "Intensity of propagated field")

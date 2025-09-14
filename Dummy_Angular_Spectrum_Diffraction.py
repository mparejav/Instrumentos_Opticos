import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   

# Dummy code to try to generate the diffraction pattern using the Angular Spectrum Method of a circular aperture
# Will try to emulate the structure given in class

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
-> 2.) Calculate A[p,q,0] - the angular spectrum at z=0 using FFT
-> 3.) Calculate A[p,q,z] - the angular spectrum at distance z using the Transfer Function
-> 4.) Calculate U[n,m,z] - the output field at distance z using inverse FFT
-> 5.) Re order the output field to center the zero frequency component
-> 6.) Calculate I[n,m,z] - the intensity at distance z
"""

# variables (for now they will be global, but should be passed as arguments to functions)
λ = 0  # um. Wavelength of light
Δ = 0  # um. Sampling interval in the spatial domain
Δf = 0 # um^-1. Sampling interval in the frequency domain
z = 0 # um. Propagation distance
N = 0 # Number of samples per side of the square grid
f_Nyquist = 0 # um^-1. Nyquist frequency. Maximum frequency that can be represented without aliasing
f_max = 0 # um^-1. Maximum frequency in the frequency domain
L = 0 # um. Physical size of the grid in the spatial domain

# Will be given values to this variables throughout the code making sure they are coherent with theorems
L = 2048 # um. Physical size of the grid in the spatial domain
λ = 0.5 # um. Wavelength of light
N = 4096 # Number of samples per side of the square grid (FOR NOW, sensaciones)

# -> We know that L = N * Δ  -> Δ = L/N ; 
Δ = L / N  # um. Sampling interval in the spatial domain

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
"""
# We'll create a transmitance of a circular aperture 
radius = 100 # um. Radius of the circular aperture

# Physical coordinates in the spatial domain
x = np.linspace(-L/2, L/2, N, endpoint=False) # start, stop, number of samples. Avoid duplicating the endpoint
y = np.linspace(-L/2, L/2, N, endpoint=False)

# Generate input field -  U(n,m,0)
U_0 = np.zeros((N, N), dtype=np.complex128)
X, Y = np.meshgrid(x, y)    # Create meshgrid
U_0 = np.where(X**2 + Y**2 <= radius**2, 1, 0) # Circular aperture transmitance

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

# For now, we will generate a padding to ensure our DFT doesnt have aliasing

padding_factor = 4 # Increase this factor to increase padding
N_padded = N * padding_factor  # New size for FFT with padding

# Calculate the FFT of the input field
A_0 = np.fft.fft2(U_0, s=(N_padded, N_padded))  # FFT2 computes the 2-dimensional discrete Fourier Transform
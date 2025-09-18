import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   

# Dummy code to generate the diffraction pattern using the Angular Spectrum Method of a circular aperture
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
# Parameters definition
λ = 0.5  # um. Wavelength of light
Propagation_Distance = 10 # um. Propagation distance
k = 2 * np.pi / λ  # um^-1. Wavenumber

N = 2048 # Number of samples per side of the square grid (FOR NOW, sensaciones)
L = 600 # um. Physical size of the grid

#Δ = 5 # um. Sampling interval in the spatial domain; L = N * Δ

# Sampling parameters
Δ = L / N  # um. Sampling interval in the spatial domain
Δf = 1 / Δ  # um^-1. Sampling interval in the frequency domain. Spectral discretization
M = 1/(λ*Δf) # Number of samples to represent the signal per axis 

# Conditions to asure proper sampling
f_max = M*Δf  # um^-1. Maximum spatial frequency
z_max = M * Δ**2 / λ # um. Maximum propagation distance in which angular spectrum method is well sampled
f_Nyquist = 1/(2*Δ)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented

if(Propagation_Distance > z_max):
    print("Exceded maximum propagation distance for proper sampling in the angular spectrum method.")
    #Propagation_Distance = z_max
    print("Propagation distance set at:", Propagation_Distance)

if(N < 2*M):
    print("Not enough samples to avoid overlapping with the circular convolutions")
    #N = 2*M
    print("Number of samples set at:", N)

if(f_max > f_Nyquist):
    print("Warning Nyquist condition not met")

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
"""
# We'll create a transmitance of a circular aperture 
radius = 10 # um. Radius of the circular aperture

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

padding_factor = 1 # Increase this factor to increase padding
N_fft = N * padding_factor  # New size for FFT with padding. Digital padding.

# Calculate the FFT of the input field
A_0 = (Δ**2) * np.fft.fft2(U_0, s=(N_fft, N_fft))  # FFT2 computes the 2-dimensional discrete Fourier Transform

"""
# -> 3.) Calculate A[p,q,z] - the angular spectrum at distance z using the Transfer Function
"""

A_z = np.zeros((N_fft, N_fft), dtype = np.complex128) # Initialize A[p,q,z]. Angular spectrum at distance z

# Arrays of spatial frequencies 
fx = np.fft.fftfreq(N_fft, d = Δ)   # frequencies along x
fy = np.fft.fftfreq(N_fft, d = Δ)   # frequencies along y

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

# Calculate the Transfer Function
Transfer_Function = np.exp(1j * Propagation_Distance * k * np.sqrt(1-(λ*FX)**2-(λ*FY)**2))

"""
DUDA: Debo shiftear Transfer Function? Según como esta definido quedan ordenados de la misma forma.
"""

# Calculate A[p,q,z] using the Transfer Function
A_z = A_0 * Transfer_Function # Element-wise multiplication

"""
-> 4.) Calculate U[n,m,z] - the output field at distance z using inverse FFT
"""

U_z = (Δf**2) * np.fft.ifft2(A_z)  # Inverse FFT to get the output field at distance z

#U_z = U_z[:N, :N]  # Crop to original size N×N if padding was used

#U_z = np.fft.fftshift(U_z)  # Shift zero frequency component to the center of the spectrum

"""
-> 5.) Calculate I[n,m,z] - the intensity at distance z
""" 

I_z = np.abs(U_z)**2  


def plot_fields(U_0, I_z, x, y, title0 = "Aperture", titlez = "Intensity field I_z"):
    """
    Plot input field Aperture and propagated output field I_z.
    The axes are set according to the physical coordinates (x, y).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Input field
    im0 = axes[0].imshow(np.abs(U_0)**2, 
                         cmap="inferno", 
                         extent=[x[0], x[-1], y[0], y[-1]])
    axes[0].set_title(title0)
    axes[0].set_xlabel("x [um]")
    axes[0].set_ylabel("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Output field
    im1 = axes[1].imshow(I_z, 
                         cmap="inferno", 
                         extent=[x[0], x[-1], y[0], y[-1]])
    axes[1].set_title(titlez)
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
  
    plt.tight_layout()
    plt.show()


plot_fields(U_0, I_z, x, y, title0 = "Transmitance", titlez = "Intensity of propagated field")

print("Done")
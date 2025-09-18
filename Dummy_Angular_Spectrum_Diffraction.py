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

# Parameters definition
λ = 0.5  # um. Wavelength of light
z = 100 # um. Propagation distance
k = 2 * np.pi / λ  # um^-1. Wavenumber
Δ = 5 # um. Sampling interval in the spatial domain; L = N * Δ
N = 1024 # Number of samples per side of the square grid (FOR NOW, sensaciones)



# Sampling parameters
L = N * Δ  # um. Physical size of the grid
Δf = 1 / L  # um^-1. Sampling interval in the frequency domain
M = 1/(λ*Δf) # Number of samples to represent the signal per axis 
f_max = M*Δf  # um^-1. Maximum frequency in the frequency domain

z = M * Δ**2 / λ
print("z should be less than:", z, "um")

f_Nyquist = 1/(2*Δ)  # um^-1. Nyquist frequency
print(f_Nyquist, "um^-1 is the Nyquist frequency")

print("Δx:", Δ, "um")

# Constraints from sampling theorems

if(N < 2*M):
    print("Warning: Increase number of samples N")
    print("Current M:", M)
    print("Current N:", N)

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
"""
# We'll create a transmitance of a circular aperture 
radius = 40 # um. Radius of the circular aperture

# Physical coordinates in the spatial domain
x = np.linspace(-L/2, L/2, N, endpoint=False) # start, stop, number of samples. Avoid duplicating the endpoint
y = np.linspace(-L/2, L/2, N, endpoint=False)

# Generate input field -  U(n,m,0)
#U_0 = np.zeros((N, N), dtype=np.complex128)
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

#padding_factor = 2 # Increase this factor to increase padding
#N_fft = N * padding_factor  # New size for FFT with padding. Digital padding.

# Calculate the FFT of the input field
A_0 = (Δ**2) * np.fft.fft2(U_0)  # FFT2 computes the 2-dimensional discrete Fourier Transform

"""
# -> 3.) Calculate A[p,q,z] - the angular spectrum at distance z using the Transfer Function
"""

A_z = np.zeros((N, N), dtype=np.complex128) # Initialize A[p,q,z]. Angular spectrum at distance z

# Arrays of spatial frequencies 
fx = np.fft.fftfreq(N, d=Δ)   # frequencies along x
fy = np.fft.fftfreq(N, d=Δ)   # frequencies along y

"""
np.fft.fftfreq:
---------------
- Generates the discrete frequency bins associated with an FFT of length n.
- Input:
    n : number of samples (FFT size, must match the array length or padded length)
    d : sample spacing in the original domain (Δx or Δt)
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
Transfer_Function = np.exp(1j*z*k * np.sqrt(1-(λ*FX)**2-(λ*FY)**2))

Transfer_Function = np.fft.fftshift(Transfer_Function)  # Shift zero frequency to center

# Calculate A[p,q,z] using the Transfer Function
A_z = A_0 * Transfer_Function # Element-wise multiplication


"""
-> 4.) Calculate U[n,m,z] - the output field at distance z using inverse FFT
"""

U_z = (Δf**2) * np.fft.ifft2(A_z)  # Inverse FFT to get the output field at distance z

#U_z = U_z[:N, :N]  # Crop to original size N×N if padding was used


"""
-> 5.) Calculate I[n,m,z] & I[n,m,0] : the intensity at distance z and at z = 0
"""

I_z = np.abs(U_z)**2  # Intensity is the magnitude squared of the field  

###### Verification of sampling theorems ######


""" Now we will try to graph the results """

def plot_fields(U_0, I_z, x, y, title0 = "Input Field U_0", titlez = "Output Field I_z"):
    """
    Plot input field U_0 and propagated output field _z.
    Both fields are shown as intensities |U|^2 for visualization.
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


plot_fields(U_0, I_z, x, y, title0="Transmitance", titlez="Intensisty of propagated field")

print("Done")
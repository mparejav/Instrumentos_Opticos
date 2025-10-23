import numpy as np    
from Miscelanea import *

# Recuperando a Paco

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
N_x = 1280 # Number of samples in x direction
N_y = 1024 # Number of samples in y direction

L_x = N_x * Δ  # um. Physical size of the sensor grid in x direction
L_y = N_y * Δ  # um. Physical size of the sensor grid in y direction
print("Physical size of the grid L_x:", L_x/1000,"x", L_y/1000, "mm")

# Setup parameters
z = 10000 # um. Propagation distance 
z_mm = z / 1000 # mm
print("Propagation distance set at:", z_mm, "mm")

# Sampling parameters
Δf_x = 1 / (Δ*N_x)  # um^-1. Sampling interval in the frequency domain. Spectral discretization 
Δf_y = 1 / (Δ*N_y)  # um^-1.

# Conditions to asure proper sampling
z_max = (N_y * Δ**2) / λ # um. Maximum propagation distance in which angular spectrum method is well sampled. We'll take the smaller N to be safe

if(z > z_max):
    #print("Exceded maximum propagation distance for proper sampling in the angular spectrum method.")
    #Propagation_Distance = z_max
    #print("Propagation distance set at:", Propagation_Distance)
    pass
    
print (z_max, "um is the maximum propagation distance for proper sampling in the angular spectrum method.")

"""
-> 1.) Take, or generate, U[n,m,0] - the input field at z=0
"""
# Create physical coordinates centered at 0
x = np.linspace(-L_x/2, L_x/2, N_x, endpoint = False)
y = np.linspace(-L_y/2, L_y/2, N_y, endpoint = False)
X, Y = np.meshgrid(x, y)

Path = select_image_path(option = 6) # Options 6 to 10 are Paco images

# Load image at its native resolution
img = Image.open(Path).convert('L')  # grayscale (0–255)
img_array = np.array(img)  # shape will be (3000, 4000) for the expected image

# Normalize grayscale values to [0,1] → transmittance mask
Intensity_field = img_array.astype(float) / 255.0  

U_0 = load_image('Images/Rochi_square.png', N)


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

Graph_Mask_and_Field_Angular_Spectrum(U_0, I_z, x, y, contrast_limit = 0.9, title_input = "Transmitancia", title_output = "Intensidad del campo propagado")

import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   

#Dummy code to try to generate the difraction pattern using The Fresnel transformation method

"""
1. Take or generate the input field as U [n,m,0]
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the spherical phase input term
3. Calculate U'' [n,m,z] with FFT 
4. Calculate U [n,m,z] adding the spherical phase output terms
5. Re organize U [n,m,z] using shift
6. Calculate the intensity to each field U_0 and U_z
"""

# Parameters definition
λ = 0.5  # um. Wavelength of light
z = 500 # um. Propagation distance
k = 2 * np.pi / λ  # um^-1. Wavenumber
L = 800 # um. Physical size of the grid in the spatial domain

N = 4096 # Number of samples per side of the square grid (FOR NOW, sensaciones)

# Sampling parameters
Δf = 1 / L  # um^-1. Sampling interval in the frequency domain
M = 1/(λ*Δf) # Number of samples to represent the signal per axis 
f_max = M*Δf  # um^-1. Maximum frequency in the frequency domain
Δ = L / N  # um. Sampling interval in the spatial domain; L = N * Δ

# Constraints from sampling theorems

if(N < 2*M):
    print("Warning: Increase number of samples N")
    print("Current M:", M)
    print("Current N:", N)


"""1. Take or generate the input field as U [n,m,0]
"""
#Creating an optic field, in this case the field is a circular slit
radius = 200 #um. The radius of the circle

#Creating the coordinates of the space

x = np.linspace (-L/2, L/2, N, endpoint =False)
y = np.linspace (-L/2, L/2, N, endpoint=False)

#Generate input field U [n,m,0]

U_0 = np.zeros ((N,N), dtype=np.complex128)
X,Y = np.meshgrid (x,y)
U_0 = np.where(X**2 + Y**2 < radius**2, 1, 0) #Circular aperture transmitance

"""
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the parabolic phases term
"""

#Firstly we are going to define the DFT kernel 


kernel = np.exp((1j*k/(2*z))*((X)**2+ (Y)**2))
U_1 = U_0*kernel

"""
3. Calculate U'' [n,m,z] with FFT 
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
N_fft = N * padding_factor  # New size for FFT with padding. Digital padding

# We are calculating the fft for U_1
U_2 =np.fft.fftshift(np.fft.fftshift(np.fft.fft2(U_1, s=(N_fft, N_fft))))

"""
4. Calculate U [n,m,z] adding the spherical phase output terms
"""

#We create the spherical phase output terms

sphericalOutput = kernel*np.exp(1j*k*z)/(1j*λ*z)

#We need to pass from Frequences Domain to space Domain
U_3 = np.fft.fftshift(np.fft.ifft(U_2))
#We multiply the U_2 per sphericalOutput term
U_z =sphericalOutput*U_3

"""
5. Re organize U [n,m,z] using shift
"""

"""
6. Calculate the intensity to each field U_0 and U_z
"""

I_z = np.abs(U_z)**2  # Intensity is the magnitude squared of the field

I_0 = np.abs(U_0)**2  # Intensity at z = 0

###### Verification of sampling theorems ######

""" Now we will try to graph the results """

def plot_fields (I_0, I_z, x, y, title0 = "Input field I_0", titlez = "Output field I_z"):
    """
    Plot input field I_0 and propagated output field Uz.
    Both fields are shown as intensities |U|^2 for visualization.
    The axes are set according to the physical coordinates (x, y).
    """
    
    fig, axes = plt.subplots (1,2,figsize = (10,4))
    
    #Input field
    im0 = axes[0].imshow(np.abs(I_0)**2, cmap ="inferno",extent= [x[0], x[-1],y[0], y[-1]])
    axes[0].set_title(title0)
    axes[0].set_xlabel ("x [um]")
    axes[0].set_ylabel ("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    #Output field
    
    im1 = axes[1].imshow(I_z, cmap="inferno", extent=[x[0], x[-1], y[0], y[-1]])
    axes[1].set_title(titlez)
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
  
    plt.tight_layout()
    plt.show()
    
plot_fields(I_0, I_z, x, y, title0="Input Field", titlez="Propagated Field")

print("Done")
    







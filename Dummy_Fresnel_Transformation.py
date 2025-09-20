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
z = 10000# um. Propagation distance
k = 2 * np.pi / λ  # um^-1. Wavenumber
#Δ_0  = 5 # um. Sampling interval in the output field; L = N * Δ
L_0 = 800 # um. Physical size of the grid in the input field
N = 2048
Δ_0 = L_0/N
# Sampling parameters
#N = int(L_0/Δ_0)  # Number of samples per side of the square grid 
Δ_f = 1 /L_0 #um^-1. sampling interval in the frequences domain 
Δ_1 =λ*z*Δ_f  # um. Sampling interval in the input field
L_1 = N* Δ_1 #um. Physical size of the grid in the output field
M = 1/(λ*Δ_f) # Number of samples to represent the signal per axis 
z_min = (N * Δ_0**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1/(2*Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented


# Constraints from sampling theorems
if(N < 2*M):
    print("Warning: Increase number of samples N")
    print("Current M:", M)
    print("Current N:", N)
  
if (z<z_min):
    print("Warning: Increase z")
    print("Current z_min:", z_min, "um")
    
  
"""1. Take or generate the input field as U [n,m,0]
"""
#Creating an optic field, in this case the field is a square slit
width = 5 #um. The radius of the circle

#Creating the coordinates of the space

x_0 = np.linspace (-L_0/2, L_0/2, N)
y_0 = np.linspace (-L_0/2, L_0/2, N)

#Generate input field U [n,m,0]
X_0,Y_0 = np.meshgrid (x_0,y_0)
U_0 = np.where((X_0**2)+(Y_0**2)<=width**2, 1, 0) #Circular aperture transmitance

"""
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the parabolic phases term
"""

#Firstly we are going to define the DFT kernel 


sphericalInput = np.exp((1j*k/(2*z))*((X_0)**2+ (Y_0)**2))
U_1 = U_0*sphericalInput

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

# We are calculating the fft for U_1
U_2 = np.fft.fftshift((Δ_0**2)*np.fft.fft2(U_1))

"""
4. Calculate U [n,m,z] adding the spherical phase output terms
"""
#We creat the coordinates for our propagated field
x_1 = np.linspace (-L_1/2, L_1/2, N)
y_1 = np.linspace (-L_1/2, L_1/2, N)

X_1,Y_1 = np.meshgrid (x_1,y_1)

#We create the spherical phase output terms

sphericalOutput =(np.exp(1j*k*z)/(1j*λ*z))*np.exp((1j*k/(2*z))*((X_1)**2+ (Y_1)**2))

#We multiply the U_2 per sphericalOutput term
U_z =sphericalOutput*U_2

"""
5. Re organize U [n,m,z] using shift
"""

"""
6. Calculate the intensity to each field U_0 and U_z
"""

I_z = np.abs(U_z)**2  # Intensity is the magnitude squared of the field

#This is for get a logaritmic graphics of output field's intensity
epsilon = 1e-6  # para evitar log(0)
I_log = np.log10(I_z + epsilon)


I_0 = np.abs(U_0)**2  # Intensity at z = 0

###### Verification of sampling theorems ######

""" Now we will try to graph the results """

def plot_fields (I_0, I_z, x_0, y_0,x_1,y_1, title0 = "Input field I_0", titlez = "Output field I_z"):
    """
    Plot input field I_0 and propagated output field Uz.
    Both fields are shown as intensities |U|^2 for visualization.
    The axes are set according to the physical coordinates (x, y).
    """
    
    fig, axes = plt.subplots (1,2,figsize = (10,4))
    
    #Input field
    im0 = axes[0].imshow(I_0, cmap ="inferno",extent= [x_0[0], x_0[-1],y_0[0], y_0[-1]])
    axes[0].set_title(title0)
    axes[0].set_xlabel ("x [um]")
    axes[0].set_ylabel ("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    #Output field
    
    im1 = axes[1].imshow(I_log, cmap="inferno", extent=[x_1[0], x_1[-1], y_1[0], y_1[-1]])
    axes[1].set_title(titlez)
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
  
    plt.tight_layout()
    plt.show()
    
plot_fields(I_0, I_z, x_0, y_0,x_1,y_1, title0="Input Field", titlez="Propagated Field")

print("Done")
    







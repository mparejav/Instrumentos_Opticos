import numpy as np 
import matplotlib.pyplot as plt  
from Miscelanea import *
from Analytical_Comparation import *


#In this module we will use the Fresnel transformation method with the parameters of ABCD matrix

"""

1. Take or generate the input field as U [n,m,0]
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the spherical phase input term
3. Calculate U'' [n,m,z] with FFT 
4. Calculate U [n,m,z] adding the spherical phase output terms
5. Calculate the intensity to each field U_0 and U_z
6. Functions for analytical solutions
"""

# Light parameters
λ = 0.633  # um. Wavelength of light (He-Ne laser)
k = 2 * np.pi / λ  # um^-1. Wavenumber

# Lens parameters
f = 500000 #um. Focal length of the lens 

#Sensor parameters CAM1 
Δ_1 = 3.8 # um. The size of the pixel in the sensor
M_x = 4640 # Number of pixels in the x axis for the sensor
M_y = 3506 # Number of pixels in the y axis for the sensor

#Input sampling field parameters
Δ_fx = 1 / (M_x * λ) #um^-1. sampling interval in the frequences domain in axis fx
Δ_fy = 1 / (M_y * λ) #um^-1. sampling interval in the frequences domain in axis fy
L_0x = 1 / Δ_fx # um. Physical size of the input field grid in axis x
L_0y = 1 / Δ_fy  # um. Physical size of the input field grid in axis y

N = 1024 # Number of samples per side of the  grid 
Δ_0 = (λ*f)/(Δ_1*N)  # um. Sampling interval in the spatial domain.


z_min = (N * Δ_0**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented    



# Graph parameters
Cut_Factor = 40 # % Porcentage cap graph


# Constraints from sampling theorems

if(f < z_min):
    print("Not enough propagation distance for proper sampling in the Fresnel transformation method.")
    print("z_min = ", z_min, "um")
    Propagation_Distance = z_min
    print("Propagation distance set at:", Propagation_Distance)
    
print("Propagation distance z:", z, "um")
    
"""
1. Take or generate the input field as U [n,m,0]
"""
#Creating the coordinates of the space centered at 0
x_0 = np.linspace (-L_0x/2, L_0x/2, N, endpoint = False)
y_0 = np.linspace (-L_0y/2, L_0y/2, N, endpoint = False)
X_0,Y_0 = np.meshgrid (x_0,y_0)

radius = 20 #um. Radius of the circle for the aperture function
length = 40 #um. Length of the rectangle for the aperture function

# Generating the aperture function. Uncomment the one you want to use

#U_0 = circle(radius, X_0, Y_0)
#U_0 = rectangle(20, 1500, X_0, Y_0)
#U_0 = vertical_slit(20, X_0, Y_0)
#U_0 = horizontal_slit(40, X_0, Y_0)
#U_0 = cross_mask(80,80,60,40,60,20,N,X,Y)
U_0 = load_image('Images\Paco.png', N)  # Sometimes its .png and sometimes .jpg
#U_0 = Ronchi_mask(lines_per_mm, X_0, Y_0)  # Ronchi grating 

"""
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the parabolic phases term
"""

#Firstly we are going to define the DFT kernel 
SphericalInput = np.exp(((1j*k)/(2*z))*((X_0)**2+ (Y_0)**2))
U_1 = U_0 * SphericalInput

"""
3. Calculate U'' [n,m,z] with FFT 
"""
              
U_2 = np.fft.fftshift(np.fft.fft2(U_1)) * (Δ_0**2)

# We are calculating the fft for U_1
#U_2 = np.fft.fftshift((Δ_0**2)*np.fft.fft2(U_1, s = (N_padded, N_padded)))  # FFT with padding

"""
4. Calculate U [n,m,z] adding the spherical phase output terms
"""
#We creat the coordinates for our propagated field
#Creating the coordinates of the space centered at 0
fx = np.fft.fftshift(np.fft.fftfreq(N, d=Δ_0))   # cycles/um
fy = fx.copy()
Xf, Yf = np.meshgrid(fx, fy)
x_1 = λ* z * fx
y_1 = λ * z * fy
X_1, Y_1 = np.meshgrid(x_1, y_1)


# Creating the radial coordinates that will be used for the Jinc pattern
R = np.sqrt(X_1**2 + Y_1**2)

#We create the spherical phase output terms
SphericalOutput =((np.exp(1j*k*z))/(1j*λ*z))*(np.exp(((1j*k)/(2*z))*(((X_1)**2)+ ((Y_1)**2))))

#We multiply the U_2 per sphericalOutput term
U_z = SphericalOutput * U_2

"""
5. Calculate I[n,m,z] - the intensity at distance z
""" 
I_z = np.abs(U_z)**2  # Intensity is the magnitude squared of the field
I_z = I_z / I_z.max()  # Normalizing to its max value

I_0 = np.abs(U_0)**2  # Intensity at z = 0   
  


""" Now we will graph the results """
plot_fields(I_0, I_z, x_0, y_0, x_1, y_1, Cut_Factor, title0 = "Transmitancia", titlez = "Intensidad del Campo propagado:\n transformada de Fresnel")
print("Done")
    
#Graph_Mask_and_Field_Angular_Spectrum(I_0, I_z, x_0, y_0, x_1, y_1, contrast_limit=0.3, title0 = "Transmitancia", titlez = "Intensidad del Campo propagado:\n transformada de Fresnel"  )







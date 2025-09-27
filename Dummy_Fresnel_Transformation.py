import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image   
from Masks import *
from scipy import special
from scipy import integrate
from scipy.special import fresnel
from scipy.integrate import quad_vec
from scipy.signal import correlate2d

#Dummy code to try to generate the difraction pattern using The Fresnel transformation method

"""
1. Take or generate the input field as U [n,m,0]
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the spherical phase input term
3. Calculate U'' [n,m,z] with FFT 
4. Calculate U [n,m,z] adding the spherical phase output terms
5. Calculate the intensity to each field U_0 and U_z
6. Functions for analytical solutions
"""

# Light parameters
λ = 0.6328  # um. Wavelength of light (He-Ne laser)
k = 2 * np.pi / λ  # um^-1. Wavenumber

# Sensor parameters (CS165MU1 Cmos sensor taken as reference)
Δ_0 = 3.475 # um. Sampling interval in the spatial domain. (Square pixel size) 3.45
N = 1024 # Number of samples per side of the square grid 
L_0 = N * Δ_0  # um. Physical size of the sensor grid (Emm...) ~ 5 mm
print("Physical size of the grid L:", L_0, "um")


"""
This parameters can be changed if we want to simulate the Paco image field (our transmitance function)
"""
#L_0 = 500 # um. Physical size of the sensor grid for paco image
#Δ_0 = L_0 / N  # um. Sampling interval in the spatial domain


# Setup parameters
z = 1500000 # um. Propagation distance

# Sampling parameters
Δ_f = 1 / L_0 #um^-1. sampling interval in the frequences domain
M = 1 / (λ * Δ_f) # Number of samples to represent the signal per axis
z_min = (N * Δ_0**2) / λ #Littlest distance z that can be well simulated with TF
f_Nyquist = 1 / (2 * Δ_0)  # um^-1. Nyquist frequency. Maximum frequency that can be accurately represented
Δ_1 = λ *z/N*Δ_0  # um. Sampling interval in the output field           
L_1 = N * Δ_1 #um. Physical size of the grid in the output field    

# Graph parameters
Cut_Factor = 90 # % Porcentage cap graph

# Talbot parameters
lines_per_mm = 10  # Ronchi grating parameter
m = 1  # Talbot length iterator
#z = Talbot_length(lines_per_mm, m)  # Calculate and print Talbot length

# Constraints from sampling theorems

if(z < z_min):
    print("Not enough propagation distance for proper sampling in the Fresnel transformation method.")
    print("z_min = ", z_min, "um")
    Propagation_Distance = z_min
    print("Propagation distance set at:", Propagation_Distance)
    
print("Propagation distance z:", z, "um")
    
"""
1. Take or generate the input field as U [n,m,0]
"""
#Creating the coordinates of the space centered at 0
x_0 = np.linspace (-L_0/2, L_0/2, N, endpoint = False)
y_0 = np.linspace (-L_0/2, L_0/2, N, endpoint = False)
X_0,Y_0 = np.meshgrid (x_0,y_0)

radius = 20 #um. Radius of the circle for the aperture function
length = 10 #um. Length of the rectangle for the aperture function

# Generating the aperture function. Uncomment the one you want to use

#U_0 = circle(radius, X_0, Y_0)
U_0 = rectangle(length, length, X_0, Y_0)
#U_0 = vertical_slit(40, X_0, Y_0)
#U_0 = horizontal_slit(40, X_0, Y_0)
#U_0 = cross_mask(80,80,60,40,60,20,N,X,Y)
#U_0 = load_image('Images\Paco.png', N)  # Sometimes its .png and sometimes .jpg
#U_0 = Ronchi_mask(lines_per_mm, X_0, Y_0)  # Ronchi grating 

"""
2. Calculate U' [n,m,0], using the function U [n,m,0] and multipling it with the parabolic phases term
"""

#Firstly we are going to define the DFT kernel 
SphericalInput = np.exp((1j*k/(2*z))*((X_0)**2+ (Y_0)**2))
U_1 = U_0 * SphericalInput

"""
3. Calculate U'' [n,m,z] with FFT 
"""

# We are calculating the fft for U_1
U_2 = np.fft.fftshift((Δ_0**2)*np.fft.fft2(U_1))

"""
4. Calculate U [n,m,z] adding the spherical phase output terms
"""
#We creat the coordinates for our propagated field
fx = np.fft.fftshift(np.fft.fftfreq(N, d=Δ_0))          # cycles / um
fy = fx.copy()
FX, FY = np.meshgrid(fx, fy)

x_1 = λ * z * fx                         # um
y_1 = λ * z * fy                         # um
X_1, Y_1 = np.meshgrid(x_1, y_1)  

# Creating the radial coordinates that will be used for the Jinc pattern
R = np.sqrt(X_1**2 + Y_1**2)

#We create the spherical phase output terms
SphericalOutput =(np.exp(1j*k*z)/(1j*λ*z))*np.exp((1j*k/(2*z))*((X_1)**2+ (Y_1)**2))

#We multiply the U_2 per sphericalOutput term
U_z = SphericalOutput * U_2

"""
5. Calculate I[n,m,z] - the intensity at distance z
""" 
I_z = np.abs(U_z)**2  # Intensity is the magnitude squared of the field
I_z = I_z / I_z.max()  # Normalizing to its max value

I_0 = np.abs(U_0)**2  # Intensity at z = 0   
  
"""
6. Functions for analytical solutions
"""

"""
In this part we are creating the anaytical solution for the circular aperture using the Jinc function
"""

#Function for generate the analytical solution of the circular aperture
def U_of_r(r):
    # we are integreting from 0 to the radius of the aperture
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
    # Factor común
    sqrt_term = np.sqrt(k/(2*z))
    # Definir u+ y u-
    u_plus  = np.sqrt(2/np.pi) * (sqrt_term*a + sqrt_term*coord)
    u_minus = np.sqrt(2/np.pi) * (-sqrt_term*a + sqrt_term*coord)

    # Fresnel en u+ y u-
    C_plus, S_plus = fresnel(u_plus)
    C_minus, S_minus = fresnel(u_minus)

    deltaC = C_plus - C_minus
    deltaS = S_plus - S_minus

    prefac = np.sqrt(np.pi*z/k) * np.exp(-1j*k*coord**2/(2*z))
    return prefac * (deltaC + 1j*deltaS)

"""
We will call these functions just when it is needed
"""

U_x = I_axis(X_1, length/2, z, k)
U_y = I_axis(Y_1, length/2, z, k)
U_sinc = (np.exp(1j*k*z) / (1j*λ*z)) * np.exp(1j*k*(X_1**2+Y_1**2)/(2*z)) * U_x * U_y
I_sinc = np.abs(U_sinc)**2
I_sinc = I_sinc / I_sinc.max()



#U_Jinc = U_of_r(R)
#I_Jinc = np.abs(U_Jinc)**2
#I_Jinc = I_Jinc / I_Jinc.max() * I_z.max()  # Normalizing to the same max value as I_z

"""
We will calculate the correlation between the numerical and analytical solutions
"""
#This is for the circular aperture
#corr_Jinc = correlate2d(I_z, I_Jinc, mode='same')
#print("Max correlation Jinc:", np.max(corr_Jinc)) 


#This is for the square aperture
#corr_sinc = correlate2d(I_z, I_sinc, mode='same')
#print("Max correlation Sinc:", np.max(corr_sinc))
          


""" Now we will graph the results """
plot_fields(I_0, I_z, x_0, y_0, x_1, y_1, Cut_Factor, title0 = "Input field I_0", titlez = "Output field I_z")

print("Done")
    







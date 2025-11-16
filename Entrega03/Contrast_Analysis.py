import numpy as np  
from Miscelanea import *
from Difraction_Implementation_Of_Matrix import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

"""
Parameteres of the optical system with variable pupile
"""
# Light parameters
λ = 0.533  # um. Wavelength of light

# Microscope objetive parameters
NA = 0.5 # Numerical aperture
M = 20  # Magnification

#Tube lens parameteres
fTL = 200000 #um. Focal distance of TL

# Objective parameters
fMO = fTL/M #um. Focal length of the lens (microscope objective) 

#Propagation distance for the first difractive implementation
d_0 = 2*fMO

#Propagation distance for the second difractive implementation
d_1 = 2*fTL

Rmax = (NA/λ)*fMO  # um. Maximum radius of the pupile

"""
1. Parameters of the object plane
We will define the coordinates of the propagated fields with the given parameters 
"""
N = 720 # Number of samples in the x and y axis for the object plane
L_3 = 390 # um. Physical size of the grid in the object plane in x and y axis
Δ_3 = L_3 / N  # um. Sampling interval in the output field

#Crop_size 
crop_size = N/M

# Creating the coordinates for the object plane
x_3, y_3, X_3, Y_3 = coordinates (L_3, L_3, N,N)

"""
2. Parameters of the propagated field towards the pupile
"""
Δ_2  = (λ *2*d_0)/(N*Δ_3)  # um. Sampling interval in the pupil plane       
L_2 = N * Δ_2 #um. Physical size of the grid in the pupil plane 
  
#Creating the coordinates for the first propagation
x_2, y_2, X_2, Y_2 = coordinates (L_2, L_2, N,N)

"""
3. Parameters of the equivalent sensor plane
"""
Δ_1  = (λ *2*d_1)/(N*Δ_2)  # um. Sampling interval in the output field    
L_1 = N * Δ_1 #um. Physical size of the grid

#Creating the coordinates for the output field
x_1, y_1, X_1, Y_1 = coordinates (L_1, L_1, N,N)

"""
4. Creating the magnificated coordinates
"""
L_magnificated = L_1*M
x_magnificated,y_magnificated, X_magnificated, Y_magnificated = coordinates(L_magnificated, L_magnificated, N, N)

"""
Creating the parameteres for the matrix system, in the first propagation towards The transmitance
In the second propagation towards the sensor CAM1, the parameters ABCD will be same as here
"""

A1, B1, C1,D1 = transferMatrix_Propagation_Lens_Propagation(fMO,fMO)

"""
Creating the input field and the output field when it is propagated to the transmitance M1
"""

U_0 = load_complex_array()


#Calculating the output field with the diffractive formulation
#Here we have the spectrum of U_0
U_beforepupile = difractive_formulation (d_0,U_0,A1,B1,D1,λ,Δ_3,Δ_3,X_3,Y_3,X_2,Y_2)


Rin_min = Rmax*0.1

# Arreglos donde guardaremos resultados
radios = np.linspace(Rin_min, Rmax, 10)              # o lo que quieras
transmitancias = np.linspace(0.0, 1.0, 10, endpoint=False)            # 0.0, 0.1, ..., 1.0

CRMS_map = np.zeros((len(radios), len(transmitancias)))
 
# Barrido doble
for i, Rin in enumerate(radios):

    for j, T in enumerate(transmitancias):

        # ---- 1. Construcción del anillo en el plano de Fourier ----
        Pupile = Variable_Radious_Transmitance(Rmax, Rin, T, X_2, Y_2)

        #Multiplying the field before the transmitance by the transmitance function xd
        U_afterpupile = U_beforepupile * Pupile

        A2, B2, C2,D2 = transferMatrix_Propagation_Lens_Propagation(fTL,fTL)
    
        #Calculating the output field with the diffractive formulation
        U_CAM1 = difractive_formulation(d_1,U_afterpupile,A2,B2,D2,λ,Δ_2,Δ_2,X_2,Y_2,X_magnificated, Y_magnificated)

        #Organizing the output field at the sensor CAM1 with shifting
        U_CAM1 = np.fft.fftshift(U_CAM1)

        """
        At this part we calculate for the intensities of the field in the CAM1
        """

        #Intensity at the sensor CAM1
        I_CAM1 = np.abs(U_CAM1)**2

        # ---- 5. Calcular media e RMS ----
        I_mean = np.mean(I_CAM1)
        sigma = np.sqrt(np.mean((I_CAM1 - I_mean)**2))

        # Contraste RMS normalizado
        CRMS = sigma / I_mean if I_mean != 0 else 0

        # ---- 6. Guardar resultado ----
        CRMS_map[i, j] = CRMS

        print(f"CRMS for radius {Rin:.3f} and transmitance {T:.1f}: {CRMS:.6f}")
        

# Normalize radius values to percentage of Rmax
radios_pct = (radios / Rmax) * 100

plt.figure(figsize=(7,6))

# Mapa de calor
plt.imshow(CRMS_map, 
           origin='lower', 
           aspect='auto',
           extent=[transmitancias[0], transmitancias[-1],
                   radios_pct[0], radios_pct[-1]], cmap='inferno')

plt.colorbar(label='CRMS')

plt.xlabel('Transmitancia (T)')
plt.ylabel(r'Radio interno (% de Rmax)')
plt.title('Mapa de contraste RMS')

plt.show()
        
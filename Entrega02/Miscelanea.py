import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''
This module provides functions to create various types of masks or apertures. Some are created analytically, 
while others can be loaded from image files.

All masks are generated as a square matrix of size size x size.
Conventional figures are plotted following symmetrical geometries centered at the origin (0,0).
All dimensions are specified in microns (um).
'''
# Function that generates a circular aperture centered in the grid.
def circle(radius_um, X, Y):
    mask = np.where(X**2 + Y**2 <= radius_um**2, 1, 0)
    return mask

# Function that generates a rectangular aperture centered.
def rectangle(width_um, height_um, X, Y):
    mask = np.where((np.abs(X) <= width_um // 2) & (np.abs(Y) <= height_um // 2), 1, 0)
    return mask

# Function that generates a vertical slit centered.
def vertical_slit(width_um, X, Y):
    mask = np.where(np.abs(X) <= width_um // 2, 1, 0)
    return mask

# Function that generates a horizontal slit centered.
def horizontal_slit(width_um, X, Y):
    mask = np.where(np.abs(Y) <= width_um // 2, 1, 0)
    return mask

"""
Function that loads a grayscale image and converts it into a binary mask.
The image must have a black background (obstacle) and white figures (aperture).

Returns:
    _type_: A binary numpy array (0 and 1) representing the mask.
"""
def load_image(image_path, Number_of_Samplesx, Number_of_Samplesy):
    img = Image.open(image_path).convert('L') # Convert to grayscale
    img = img.resize((Number_of_Samplesx, Number_of_Samplesy)) # Resize to simulation matrix
    img_array = np.array(img)
    # Normalize: values >128 are considered as aperture (1), others as obstacle (0)
    mask = np.where(img_array > 128, 1, 0)
    return mask


def Intensity_Field_Diffraction_Pattern(image_path):
    """
    Load a 4000x3000 pixel image from a 5 μm/pixel sensor and returns it as 
    a grayscale array.

    Parameters
    ----------
    image_path : str
        Path to the grayscale image (expected 4000x3000 pixels from 5 μm/pixel sensor). Not completely sure of pixel size.

    Returns
    -------
    mask : ndarray (dtype float64)
        Grayscale transmittance (0 to 1) with shape (3000, 4000).
    pixel_size : float
        Physical pixel size in micrometers (5.0 μm).
    """

    # Load image at its native resolution
    img = Image.open(image_path).convert('L')  # grayscale (0–255)
    img_array = np.array(img)  # shape will be (3000, 4000) for the expected image
    
    """
    As we are working with a known image theres no need to resize it. Every pixel in the original image will be an element of the 
    array. This is highly demanding for processing, but it preserves all the information of the image. The code (Reverse_Angular_Spectrum.py) 
    assumes the known parameters.
    """
    # Normalize grayscale values to [0,1] → transmittance mask
    Intensity_field = img_array.astype(float) / 255.0  

    return Intensity_field


""""
This function plots an aperture and its correspondent diffraction pattern.
The axes are set according to the physical coordinates (x, y).
Requires as inputs two matrix: the first argument is the binary array asociated to the mask; the second one is the array for
the Intensity field of a difraction pattern propagated a distance z; the third and fourth are the spatial variables; fifth is the
"cut factor", this is a porcentage value that caps the maximum value so the funtion reduces its contrast.

Returns:
    An image with two figures.
"""
def plot_fields(Mask, Intensity_Propagated_Field, x, y, x_p, y_p, Cut_Factor, title0 = "Aperture", titlez = "Intensity field I_z"):
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Input field
    im0 = axes[0].imshow(Mask, cmap="gray", extent=[x[0], x[-1], y[0], y[-1]])
    axes[0].set_title(title0)
    axes[0].set_xlabel("x [um]")
    axes[0].set_ylabel("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Porcentage to decimal
    Cut_Factor = (Cut_Factor / 100)

    # Output field
    im1 = axes[1].imshow(Intensity_Propagated_Field, cmap="gray", extent=[x_p[0], x_p[-1], y_p[0], y_p[-1]], vmax = np.max(Intensity_Propagated_Field)*Cut_Factor)
    axes[1].set_title(titlez)
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
  
    plt.tight_layout()
    plt.show()

"""
This function plots an aperture and its correspondent diffraction pattern.
The axes are set according to the physical coordinates (x, y).
Requires as inputs two matrix: the first argument is the binary array asociated to the mask; the second one is the array for
the Intensity field of a difraction pattern propagated a distance z; the third and fourth are the spatial variables; fifth is the
"cut factor" as a decimal, this is a porcentage value that caps the maximum value so the funtion reduces its contrast.
"""
def Graph_Mask_and_Field_Angular_Spectrum(Mask, Intensity_Propagated_Field, x, y, contrast_limit, title_input = 'Aperture', title_output = 'Propagated Intensity Field'):
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Input field
    im0 = axes[0].imshow(Mask, cmap = "inferno", extent=[x[0], x[-1], y[0], y[-1]], vmax = np.max(Mask) * contrast_limit)
    axes[0].set_title(title_input)
    axes[0].set_xlabel("x [um]")
    axes[0].set_ylabel("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction = 0.046, pad = 0.04)

    """
    According to the discrete formulation the sampling intervals satisfy:

        Δ · Δf = 1 / N 

    where Δ is the spatial sampling interval and Δf is the spectral sampling interval.
    When we propagate the field using the Angular spectrum method, the process is:

        1) FFT: U(x,y,0)  →  A(fx,fy,0)
        2) Multiply by transfer function: A(fx,fy,0) → A(fx,fy,z)
        3) IFFT: A(fx,fy,z) → U(x,y,z)

    Both the FFT and the IFFT use the same number of samples (N),
    and the same relation Δ · Δf = 1/N. Therefore, the spatial sampling interval
    after propagation is preserved:

        Δ_out = Δ_in

    This means that the physical coordinate grids of the output plane are the same
    as those of the input plane.
    """
    # Output field
    im1 = axes[1].imshow(Intensity_Propagated_Field, cmap= "inferno", extent=[x[0], x[-1], y[0], y[-1]], vmax = np.max(Intensity_Propagated_Field) * contrast_limit)
    axes[1].set_title(title_output)
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    plt.colorbar(im1, ax=axes[1], fraction = 0.046, pad = 0.04)

    plt.tight_layout()
    plt.show()
    
def plot_correlation(I_0,I_numerical, I_analytical,correlation_intensity, x_0, y_0, x_1,y_1, title = "Input field", title0 = "Numerical Solution", title1 = "Analytical Solution", title2 = "Correlation Intensity"):
   
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    #Plottting the input field
    im00 = axes[0,0].imshow(I_0, cmap='inferno', extent=[x_0[0], x_0[-1], y_0[0], y_0[-1]])
    axes[0,0].set_title(title) 
    axes[0,0].set_xlabel('x [um]')
    axes[0,0].set_ylabel('y [um]')
    plt.colorbar(im00, ax=axes[0,0], fraction = 0.046, pad = 0.04)
    
    #Plotting the numerical solution
    im01 = axes[0,1].imshow(I_numerical, cmap='inferno', extent=[x_1[0], x_1[-1], y_1[0], y_1[-1]])
    axes[0,1].set_title(title0)
    axes[0,1].set_xlabel('x [um]')
    axes[0,1].set_ylabel('y [um]')
    plt.colorbar(im01, ax=axes[0,1], fraction = 0.046, pad = 0.04)
    
    #Plotting the analytical solution
    im10 = axes[1,0].imshow(I_analytical, cmap='inferno', extent=[x_1[0], x_1[-1], y_1[0], y_1[-1]])
    axes[1,0].set_title(title1)
    axes[1,0].set_xlabel('x [um]')
    axes[1,0].set_ylabel('y [um]')
    plt.colorbar(im10, ax=axes[1,0], fraction = 0.046, pad = 0.04)

    
    # Plotting the correlation intensity
    im11 = axes[1,1].imshow(correlation_intensity, cmap='inferno', extent=[x_0[0], x_0[-1], y_0[0], y_0[-1]])
    axes[1,1].set_title(title2)
    axes[1,1].set_xlabel('x [um]')
    axes[1,1].set_ylabel('y [um]')
    plt.colorbar(im11, ax=axes[1,1], fraction = 0.046, pad = 0.04)
    plt.tight_layout()
    plt.show()
  
 
"""
Generates a Ronchi ruling mask.

Args:
    lines_per_mm: Number of lines per mm (spatial frequency).
    X, Y: Meshgrids with spatial coordinates [um].
    n: Iterator for Talbot length calculation.

Returns:
    mask (2D array): Binary mask with Ronchi grating pattern.
"""
# Function that generates a Ronchi ruling mask
def Ronchi_mask(lines_per_mm, X, Y):

    # Convert lines/mm to spatial frequency in microns
    period_um = 1000 / lines_per_mm   # um per line pair (bright+dark)

    # Square wave along x-axis: 1 inside slit, 0 outside
    mask = (np.mod(X, period_um) < (period_um / 2)).astype(int)
    
    return mask

"""Calculate the Talbot length for a given Ronchi grating.

Args:
    lines_per_mm: Number of lines per mm (spatial frequency).
    n: Iterator for Talbot length calculation.
    
Returns:
    Propagation_Distance_um : Calculated Talbot length in microns.
"""
def Talbot_length(lines_per_mm, n):

    λ = 0.633  # um. Wavelength of the light source (He-Ne laser)

    # Convert lines/mm to spatial frequency in microns
    period_um = 1000 / lines_per_mm   # um per line pair (bright+dark)

    # Calculates Talbot length
    talbot_length = (n *2* period_um**2) / λ
    
    print(f"Talbot length for {lines_per_mm} lines/mm: {talbot_length:.2f} um")
    
    Propagation_Distance_um = talbot_length
    print(f"Propagation distance set at: {Propagation_Distance_um} um")

    return Propagation_Distance_um

def Paco_mask(Number_of_Samples):
    
    L = 5800 # um. Physical size of the sensor grid for paco image
    
    x = np.linspace(-L/2, L/2, Number_of_Samples, endpoint = False)
    y = np.linspace(-L/2, L/2, Number_of_Samples, endpoint = False)
    X, Y = np.meshgrid(x, y)

    Pixel_Size = L / Number_of_Samples  # um. Sampling interval in the spatial domain. (Square pixel size)

    return X, Y, Pixel_Size

# Switch case structure for image path selection in Reverse_Angular_Spectrum.py
def select_image_path(option):
    switcher = {
        1: 'Images/Trans_Cam_77mm_Luz_Trans_1_1_2pulgada.png',
        2: 'Images/Trans_Cam_111mm_Luz_Trans_1_1_2pulgada.png',
        3: 'Images/Trans_Cam_156mm_Luz_Trans_1_1_2pulgada.png',
        4: 'Images/Trans_Cam_109mm_Luz_Trans_46mm.png',
        5: 'Images/Trans_Cam_109mm_Luz_Trans_6_pulgada.png'
        }
    return switcher.get(option, "Invalid option")

#Creating the transmitance function for the mirror M1 when this is a function = 1
def transmitance_1 (L_x,L_y, X,Y):
    Transmitance = np.where((np.abs(X) <= L_x/2) & (np.abs(Y) <= L_y/2), 1, 0)
    return Transmitance

#Pad a field having a sample 
def pad (inputField, samplingField):
    N1x, N1y = inputField.shape
    N2x, N2y = samplingField.shape  # needed size
    pad_x_before = (N2x - N1x) // 2
    pad_x_after  = N2x - N1x - pad_x_before
    pad_y_before = (N2y - N1y) // 2
    pad_y_after  = N2y - N1y - pad_y_before
    inputFieldPadded = np.pad(inputField, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)), mode='constant')
    
    return inputFieldPadded

#Function for create coordinates
def coordinates (L_x, L_y, N_x, N_y):
    x = np.linspace(-L_x/2, L_x/2, N_x)  # um. x coordinates of the transmitance M1
    y = np.linspace(-L_y/2, L_y/2, N_y)  # um. y coordinates of the transmitance M1
    X, Y = np.meshgrid(x, y)
    return x,y,X,Y

#path = select_image_path(1)

#print(path)

"""
Functions to create the transfer matrices for the optical systems
"""

#We are creating the matrix for O(u,v) in terms of f
def transferMatrix_With_Reflection(f):
    #We are creating the matrix for O(u,v) in terms of f
    matrixO = [[[1, f], [0, 1]], [[1, 0], [-1/f, 1]], [[1, f], [0, 1]], 
           [[1, 0],[0, 1]],[[1, f], [0, 1]],[[1, 0], [-1/f, 1]],[[1, f], [0, 1]]]
    
    #We are multippling the matrices for transfer s(xi,eta) to O(u,v) or U(x',y')
    
    result = matrixO[0]
    for j in range(1, len(matrixO)):
        result = np.dot(result, matrixO[j])
    return result [0][0], result[0][1], result[1][0], result[1][1]

#We are creating the matrix for propagation - lens - propagation
def transferMatrix_Propagation_Lens_Propagation(f,d):
    #We are creating the matrix for U(u',v') in terms of f and d
    matrixU = [[[1, f], [0, 1]], [[1, 0], [-1/f, 1]], [[1, d], [0, 1]]]
    
    #We are multipling the matrices for transfer s(xi,eta) to O(u,v) or U(x',y')
    
    result = matrixU[0]
    for j in range(1, len(matrixU)):
        result = np.dot(result, matrixU[j])
    return result [0][0], result[0][1], result[1][0], result[1][1]
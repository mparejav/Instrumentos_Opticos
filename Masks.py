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
Function that generates a cross-shaped mask centered, with independent thicknesses in vertical and horizontal.
The cross is formed by two rectangles: one horizontal (with thickness 'e_um') and one vertical (with thickness 't_um').
The dimensions are defined in microns (um).
"""
def cross_mask(L1_um, L2_um, h1_um, h2_um, t_um, e_um, Number_of_Samples, X, Y):

    mask = np.zeros((Number_of_Samples, Number_of_Samples))

    # Horizontal arm of the cross
    cond_h = (np.abs(Y) <= e_um/2) & (X >= -L1_um - t_um/2) & (X <= L2_um + t_um/2)
    mask[cond_h] = 1

    # Vertical arm of the cross
    cond_v = (np.abs(X) <= t_um/2) & (Y >= -h2_um - e_um/2) & (Y <= h1_um + e_um/2)
    mask[cond_v] = 1

    return mask

"""
Function that loads a grayscale image and converts it into a binary mask.
The image must have a black background (obstacle) and white figures (aperture).

Returns:
    _type_: A binary numpy array (0 and 1) representing the mask.
"""
def load_image(image_path, Number_of_Samples):
    img = Image.open(image_path).convert('L') # Convert to grayscale
    img = img.resize((Number_of_Samples, Number_of_Samples)) # Resize to simulation matrix
    img_array = np.array(img)
    # Normalize: values >128 are considered as aperture (1), others as obstacle (0)
    mask = np.where(img_array > 128, 1, 0)
    return mask

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
    im0 = axes[0].imshow(Mask, cmap="inferno", extent=[x[0], x[-1], y[0], y[-1]])
    axes[0].set_title(title0)
    axes[0].set_xlabel("x [um]")
    axes[0].set_ylabel("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Porcentage to decimal
    Cut_Factor = (Cut_Factor / 100)

    # Output field
    im1 = axes[1].imshow(Intensity_Propagated_Field, cmap="inferno", extent=[x_p[0], x_p[-1], y_p[0], y_p[-1]])
    axes[1].set_title(titlez)
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("y [um]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
  
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

def Talbot_length(lines_per_mm, n):
    """Calculate the Talbot length for a given Ronchi grating.

    Args:
        lines_per_mm: Number of lines per mm (spatial frequency).
        n: Iterator for Talbot length calculation.
        
    Returns:
        Propagation_Distance_um : Calculated Talbot length in microns.
    """

    λ = 0.633  # um. Wavelength of the light source (He-Ne laser)

    # Convert lines/mm to spatial frequency in microns
    period_um = 1000 / lines_per_mm   # um per line pair (bright+dark)

    # Calculates Talbot length
    talbot_length = (n * 2*period_um**2) / λ
    
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

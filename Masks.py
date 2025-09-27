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
    im1 = axes[1].imshow(Intensity_Propagated_Field, cmap="inferno", extent=[x_p[0], x_p[-1], y_p[0], y_p[-1]], vmax = np.max(Intensity_Propagated_Field)*Cut_Factor)
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
    im0 = axes[0].imshow(Mask, cmap = "inferno", extent=[x[0], x[-1], y[0], y[-1]])
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
    talbot_length = (n * 2 * period_um**2) / λ
    
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

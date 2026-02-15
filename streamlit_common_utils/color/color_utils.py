# Imports
import colorsys
import matplotlib.pyplot as plt

# Main Functions
# Basic Dataset Functions
def get_cmap_gradient_color_point(i, n=1.0, cmap="gist_gray") -> tuple:
    '''
    Gets a color from a colormap based on the index and total number of colors

    Args:
        i (int): Index of the color
        n (int): Total number of colors
        cmap (str): Colormap name

    Returns:
        color (tuple): RGBA color tuple
    '''
    return plt.get_cmap(cmap)(1.0 * (i/n))

def combine_colors_alpha_composite(c1, c2) -> list:
    '''
    Alpha composite c1 over c2.

    Args:
        c1 (tuple/list): RGBA color 1 (should be in 0–255 or 0–1 range)
        c2 (tuple/list): RGBA color 2 (should be in 0–255 or 0–1 range)

    Returns:
        list: Combined RGBA color
    '''

    if len(c1) != 4 or len(c2) != 4:
        raise ValueError("Both colors must be RGBA (4 components)")

    # Detect alpha scale
    max_alpha = max(c1[-1], c2[-1])
    scale = 255.0 if max_alpha > 1 else 1.0

    # Normalize alphas to 0–1
    a1 = c1[-1] / scale
    a2 = c2[-1] / scale

    # Porter-Duff source-over formula
    out_alpha = a1 + a2 * (1 - a1)

    if out_alpha == 0:
        return [0] * len(c1)

    out_color = []
    for i in range(len(c1) - 1):
        c = (
            c1[i] * a1 +
            c2[i] * a2 * (1 - a1)
        ) / out_alpha
        out_color.append(c)

    # Convert alpha back to original scale
    out_color.append(out_alpha * scale)

    # If original was 0–255, round to ints
    if scale == 255.0:
        out_color = [int(round(v)) for v in out_color]

    return out_color

def generate_rainbow_colors(n) -> list:
    '''
    Generate n distinct colors in a rainbow spectrum

    Args:
        n (int): Number of colors to generate

    Returns:
        list: List of RGB color tuples
    '''
    colors = []

    for i in range(n):
        c = colorsys.hsv_to_rgb(i/n, 1.0, 1.0)
        colors.append(c)
    
    return colors
# Imports
import matplotlib.pyplot as plt

# Main Functions
# Basic Dataset Functions
def display_image_using_matplotlib(I, title=None) -> None:
    '''
    Display an image using matplotlib.

    Args:
        I: Image array (H, W, 3) in BGR or RGB format.
        title: Optional title for the image.

    Returns:
        None
    '''
    plt.imshow(I)
    plt.title(title)
    plt.show()
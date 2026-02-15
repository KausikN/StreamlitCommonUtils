# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Main Functions
# Basic Dataset Functions
def display_image_using_matplotlib(I, title=None) -> None:
    '''
    Display an image using matplotlib

    Args:
        I: Image array (H, W, 3) in BGR or RGB format
        title: Optional title for the image

    Returns:
        None
    '''
    plt.imshow(I)
    plt.title(title)
    plt.show()

def read_image(imgPath, imgSize=None, keepAspectRatio=False) -> np.ndarray:
    '''
    Read an image from a given path and resize it to a given size

    Args:
        imgPath: Path to the image file
        imgSize: Desired size (width, height) to resize the image to. If None, keeps original size.
        keepAspectRatio: Whether to keep the aspect ratio when resizing

    Returns:
        I: The read (and resized) image as a numpy array
    '''
    I = cv2.imread(imgPath)
    if not imgSize == None:
        size_original = [I.shape[0], I.shape[1]]
        if keepAspectRatio:
            if imgSize[1] > imgSize[0]:
                imgSize = (size_original[0] * (imgSize[1] / size_original[1]), imgSize[1])
            elif imgSize[0] > imgSize[1]:
                imgSize = (imgSize[0], size_original[1] * (imgSize[0] / size_original[0]))
            else:
                if size_original[1] > size_original[0]:
                    imgSize = (size_original[0] * (imgSize[1] / size_original[1]), imgSize[1])
                else:
                    imgSize = (imgSize[0], size_original[1] * (imgSize[0] / size_original[0]))
            imgSize = (int(round(imgSize[1])), int(round(imgSize[0])))
        I = cv2.resize(I, imgSize)
    return I

def save_image(imgPath, I) -> None:
    '''
    Save an image to a given path

    Args:
        imgPath: Path to save the image file
        I: Image array (H, W, 3) in RGB format to save

    Returns:
        None
    '''
    cv2.imwrite(imgPath, I)

def add_text_to_image_with_box(
    I, text, 
    start_left_top=(True, True),
    position_relative=(0.1, 0.1),
    position_absolute=(0, 0), 
    font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, padding_scale=0.01,
    bg_color=(0, 0, 0), color=(255, 255, 255), thickness_scale=1
) -> np.ndarray:
    '''
    Add text to an image at a specified position with a background box for better visibility

    Args:
        I: Image array (H, W, 3) in RGB format
        text: Text string to add to the image
        start_left_top: Tuple indicating whether the position is relative to the top-left corner (True) or bottom-right corner (False) for each dimension
        position_relative: Tuple of relative positions (between 0 and 1) for the text in the image dimensions
        position_absolute: Tuple of absolute pixel offsets to add to the calculated position
        font: OpenCV font type for the text
        font_scale: Scale factor for the font size
        padding_scale: Scale factor for the padding around the text box relative to the image size
        bg_color: Background color for the text box (RGB format)
        color: Color of the text (RGB format)
        thickness_scale: Scale factor for the thickness of the text and box borders relative to the image size

    Returns:
        I_with_text: The image with the added text and background box
    '''
    I_size = I.shape[:2]
    pos = [0, 0]
    for i in range(2):
        if start_left_top[i]:
            pos[i] = int(position_relative[i] * I_size[i]) + position_absolute[i]
        else:
            pos[i] = int((1 - position_relative[i]) * I_size[i]) + position_absolute[i]

    fontScale = font_scale * (np.max(I.shape) / 512)
    fontThickness = max(1, int(thickness_scale * (np.max(I.shape) / 512)))
    padding = [int(I.shape[0] * padding_scale), int(I.shape[1] * padding_scale)]

    text_size, _ = cv2.getTextSize(text, font, fontScale, fontThickness)
    text_w, text_h = text_size
    I_t = cv2.rectangle(I, (pos[0] - padding[0], pos[1] + padding[1]), (pos[0] + text_w + padding[0], pos[1] - text_h - padding[1]), bg_color, -1)
    I_t = cv2.putText(I_t, text, pos, font, fontScale, color, fontThickness)

    return I_t
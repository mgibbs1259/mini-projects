import os
import argparse

import cv2
import numpy as np


def load_image(img_path: str) -> np.array:
    """
    Load an image from the specified path.

    Args:
        img_path: Path to the image file.

    Returns:
        The loaded image as a NumPy array.
    """
    return cv2.imread(img_path)


def preprocess_image(image: np.array) -> np.array:
    """
    Preprocess the image for contour extraction.

    Args:
        image: Input image as a NumPy array.

    Returns:
        The preprocessed image as a NumPy array.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding to create a binary image
    _, thresholded_image = cv2.threshold(gray, 210, 235, 1)

    return thresholded_image


def extract_contours(thresholded_image: np.array) -> np.array:
    """
    Extract contours from the thresholded image. A contour is a curve joining all the 
    continuous points along the boundary of an object, which has the same color or intensity. 
    In this script, contours are used to outline the regions of interest in the image, 
    such as individual objects or shapes.

    Args:
        thresholded_image: Thresholded image as a NumPy array.

    Returns:
        The contours as a NumPy array.
    """
    contours, _ = cv2.findContours(
        thresholded_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def save_contours(
    image: np.array, 
    contours: np.array, 
    output_dir: str,
    filename: str
    ) -> None:
    """
    Save the extracted contours as individual images.

    Args:
        image: Input image as a NumPy array.
        contours: Contours to be saved.
        output_dir: Path to the output directory.
        filename: Name of the input file.

    Returns:
        None
    """
    # Calculate the image area
    image_area = image.shape[0] * image.shape[1]
    
    for i, c in enumerate(contours):
        # Get the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculate the contour area
        contour_area = cv2.contourArea(box)
        
        if image_area / 10 < contour_area < image_area * 2 / 3:
            # Extract the contour region from the image
            (x, y, w, h) = cv2.boundingRect(box)
            photo = image[y:y + h, x:x + w]
            
            output_path = os.path.join(output_dir, f"{filename}_{i + 1}.png")
            cv2.imwrite(output_path, photo)


def process_image(img_path: str, output_dir: str) -> None:
    """
    Process an image by extracting and saving its contours.

    Args:
        img_path: Path to the input image.
        output_dir: Path to the output directory for saving contours.

    Returns:
        None
    """
    # Load the image
    image = load_image(img_path)
    # Preprocess the image
    thresholded_image = preprocess_image(image)
    # Extract contours from the thresholded image
    contours = extract_contours(thresholded_image)
    # Save the extracted contours as individual images
    save_contours(image, contours, output_dir, "".join(os.path.basename(img_path).split(".")[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract individual photos from an image containing multiple photos"
        )
    parser.add_argument("input_dir", type=str, help="Path to input directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    
    args = parser.parse_args()

    for filename in os.listdir(args.input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(args.input_dir, filename)
            try:
                process_image(img_path, args.output_dir)
            except Exception as e:
                print(f"An error occurred for {img_path}: {e}")

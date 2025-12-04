"""
Signature image preprocessing module - K-means Focused
Cleans dotted paper backgrounds using K-means clustering
"""

import cv2
import numpy as np

def remove_background_noise_kmeans(image, n_clusters=3):
    """
    Removes background noise using K-means clustering
    MOST EFFECTIVE METHOD FOR DOTTED PAPER!

    Args:
        image: Grayscale image
        n_clusters: Number of clusters (3 = background + signature + transition)

    Returns:
        Cleaned binary image (black signature, white background)
    """
    # Reshape the image
    data = image.reshape((-1, 1))
    data = np.float32(data)

    # K-means clustering - parameters optimized for more stable results
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Sort centers (darkest = signature, lightest = background)
    centers = centers.flatten()
    sorted_indices = np.argsort(centers)

    # Accept the darkest cluster as signature
    signature_label = sorted_indices[0]  # Lowest intensity = darkest = signature

    # Make signature black (0), background white (255)
    result = np.where(labels.reshape(image.shape) == signature_label, 0, 255).astype(np.uint8)

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    return result

def advanced_signature_preprocessing(image, method='kmeans'):
    """
    Advanced signature preprocessing main function
    K-means method is used by default (most effective result)

    Args:
        image: Input image (BGR or grayscale)
        method: 'kmeans' (recommended and main method)

    Returns:
        Cleaned grayscale image
    """
    # Convert to grayscale if BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if method == 'kmeans':
        # K-means based background removal (MAIN METHOD)
        processed = remove_background_noise_kmeans(gray)
    else:
        # Other methods existed in the past, now only K-means is used
        print(f"Warning: '{method}' method is not supported. Using K-means.")
        processed = remove_background_noise_kmeans(gray)
    
    return processed

if __name__ == "__main__":
    # Sample usage for testing
    print("K-means Focused Signature Preprocessing Module Ready!")
    print("Usage:")
    print("from image_preprocessing import advanced_signature_preprocessing")
    print("processed_image = advanced_signature_preprocessing(image, method='kmeans')")
    print("\nDotted paper backgrounds are automatically cleaned!")

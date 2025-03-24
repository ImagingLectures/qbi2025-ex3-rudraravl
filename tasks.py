import numpy as np
from skimage.io import imread
from scipy.ndimage import gaussian_filter, median_filter

def mse(img1,img2):
    return np.mean((img1-img2)**2)

def snr_mean_std(image: np.ndarray) -> float:
    """Compute SNR as mean divided by standard deviation."""
    if np.std(image) == 0:
        return np.inf
    return np.mean(image)/np.std(image)


def snr_power_linear(img: np.ndarray) -> float:
    """Compute SNR as signal power (mean squared) over noise power (variance)."""
    if np.var(img) == 0:
        return np.inf
    return np.mean(img)**2/np.var(img)


def psnr(img: np.ndarray, max_val: float = 255.0) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) of an image.

    PSNR is defined as:

    .. math::
        PSNR = 10 \\log_{10} \\left(\\frac{\\max(I)^2}{\\sigma^2} \\right)

    where:
        - \\( \\max(I) \\) is the maximum possible pixel value (e.g., 255 for 8-bit images).
        - \\( \\sigma^2 \\) is the variance of the image, assumed to represent noise.

    Parameters:
        img (np.ndarray): Input image.
        max_val (float, optional): Maximum pixel value. Defaults to 255.

    Returns:
        float: PSNR value in decibels (dB). Returns `inf` if the variance is zero.
    """
    if np.var(img) == 0:
        return np.inf
    return 10*np.log10(max_val**2/np.var(img))


def snr_known_noise(image: np.ndarray, noise: np.ndarray) -> float:
    """Calculate the Signal-to-Noise Ratio (SNR) of an image using a known noise image.

    Parameters:
        image (np.ndarray): The input image.
        noise (np.ndarray): The noise image.

    Returns:
        float: The SNR value in dBs.

    Raises:
        ZeroDivisionError: If the sum of noise squared is zero.
    """
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        raise ZeroDivisionError("Noise power is zero.")
    return 10*np.log10(np.sum(image**2)/noise_power)


def snr_second_region_image_a() -> float:
    """
    Identify a constant region on the sample and compute the snr as the mean divided by the standard deviation.

    Returns:
        float: SNR value for the selected region.
    """
    a=imread('data/scroll.tif')
    return np.mean(a[1000:1060, 800:850])/np.std(a[1000:1060, 800:850])


def snr_image_b() -> float:
    """
    Identify a constant region outside of the sample and compute the snr as the mean divided by the standard deviation.

    Returns:
        float: SNR value for the selected region.
    """
    b = imread('data/wood.tif')
    return np.mean(b[200:250, 100:150]) / np.std(b[200:250, 100:150])


def snr_image_c() -> float:
    """
    Identify a constant region outside of the sample and compute the snr as the mean divided by the standard deviation.

    Returns:
        float: SNR value for the selected region.
    """
    c = imread('data/asphalt_gray.tif')
    # Adjust the region to find a more constant area
    sample = c[10:150, 0:150]
    return np.mean(sample) / np.std(sample)


def filter_image_gaussian_noise() -> np.ndarray:
    """
    Filter the image to reduce the noise.

    Returns:
        np.ndarray: The filtered image.
    """
    original_img=np.mean(imread('data/testpattern.png'),2)/255.
    SNR=2
    noised_image = original_img/original_img.max() + 1.0/SNR*np.random.normal(0,1,size=original_img.shape)

    recovered_img = gaussian_filter(noised_image, 3)
    return recovered_img

def filter_image_poisson_noise() -> np.ndarray:
    """
    Filter the image to reduce the noise.

    Returns:
        np.ndarray: The filtered image.
    """
    original_img=np.mean(imread('data/testpattern.png'),2)/255.
    SNR=2
    noised_image = original_img/original_img.max() + 1.0/SNR*np.random.poisson(5,size=original_img.shape)

    recovered_img = median_filter(noised_image, 5)
    return recovered_img
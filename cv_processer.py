import os
import time
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def to_grayscale_manual(img):
    h, w, c = img.shape
    gray = np.zeros((h, w), dtype=np.uint8)

    # for i in range(h):
    #     for j in range(w):
    #         b, g, r = img[i, j]
    #         gray[i, j] = int(0.114 * b + 0.587 * g + 0.299 * r)

    coeffs = np.array([0.114, 0.587, 0.299])
    gray = np.dot(img, coeffs).astype(np.uint8)

    return gray


def convolution_manual(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(img)

    # for i in range(h):
    #     for j in range(w):
    #         region = padded[i:i + kh, j:j + kw]
    #         result[i, j] = np.clip(np.sum(region * kernel), 0, 255)

    H, W = result.shape
    kh, kw = kernel.shape

    shape = (H, W, kh, kw)
    strides = padded.strides * 2  # (s0, s1, s0, s1)
    windows = as_strided(padded, shape=shape, strides=strides)

    conv = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))

    # result = np.clip(conv, 0, 255).astype(np.uint8)
    result = conv

    return result


def gaussian_kernel(size=5, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def sobel_manual(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = convolution_manual(img, Kx)
    gy = convolution_manual(img, Ky)

    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_max = grad.max()
    if grad_max > 0:
        grad = grad / grad_max * 255

    return grad.astype(np.uint8)


def sobel_manual_threshold(img, threshold=100):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = convolution_manual(img, Kx)
    gy = convolution_manual(img, Ky)

    grad = np.sqrt(gx ** 2 + gy ** 2)

    # Применяем порог: все пиксели выше threshold станут 255, остальные — 0
    grad_binary = np.where(grad >= threshold, 255, 0)

    return grad_binary.astype(np.uint8)


def main(input_dir='paintings', output_dir='filters'):
    files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    if not files:
        print("Нет изображений.")
        return

    image_path = os.path.join(input_dir, files[0])
    img = cv2.imread(image_path)
    scale = 1

    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)

    start = time.time()
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print("OpenCV resize time:", time.time() - start)
    img = img_resized

    start = time.time()
    gray_manual = to_grayscale_manual(img)
    print("Manual grayscale:", time.time() - start)

    start = time.time()
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("OpenCV grayscale:", time.time() - start)

    kernel = gaussian_kernel(5, 1)

    start = time.time()
    gauss_manual = convolution_manual(gray_manual, kernel)
    print("Manual Gaussian:", time.time() - start)

    start = time.time()
    gauss_cv = cv2.filter2D(gray_cv, -1, kernel)
    print("OpenCV Gaussian:", time.time() - start)

    start = time.time()
    sobel_img = sobel_manual(gray_manual)
    print("Manual Sobel:", time.time() - start)

    start = time.time()
    canny_cv = cv2.Canny(gray_cv, 100, 200)
    print("OpenCV Canny:", time.time() - start)

    start = time.time()
    harris_cv = cv2.cornerHarris(np.float32(gray_cv), 2, 5, 0.04)
    harris_norm = cv2.normalize(harris_cv, None, 0, 255, cv2.NORM_MINMAX)
    harris_uint8 = harris_norm.astype(np.uint8)
    print("OpenCV Harris:", time.time() - start)

    start = time.time()
    gauss_sobel = sobel_manual(gauss_manual)
    print("Gauss Sobel:", time.time() - start)

    start = time.time()
    sobel_new = sobel_manual_threshold(gray_cv,threshold=50)
    print("Gauss Sobel:", time.time() - start)

    start = time.time()
    gauss_sobel_full = sobel_manual_threshold(gauss_manual, threshold=50)
    print("Gauss Sobel:", time.time() - start)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "gray_manual.jpg"), gray_manual)
    cv2.imwrite(os.path.join(output_dir, "gray_cv.jpg"), gray_cv)
    cv2.imwrite(os.path.join(output_dir, "gauss_manual.jpg"), np.clip(gauss_manual, 0, 255).astype(np.uint8))
    # cv2.imwrite(os.path.join(output_dir, "gauss_manual.jpg"), gauss_manual)
    cv2.imwrite(os.path.join(output_dir, "gauss_cv.jpg"), gauss_cv)
    cv2.imwrite(os.path.join(output_dir, "sobel_manual.jpg"), sobel_img)
    cv2.imwrite(os.path.join(output_dir, "canny_cv.jpg"), canny_cv)
    cv2.imwrite(os.path.join(output_dir, "harris_cv.jpg"), harris_uint8)
    cv2.imwrite(os.path.join(output_dir, "Gauss_Sobel.jpg"), gauss_sobel)
    cv2.imwrite(os.path.join(output_dir, "New_Sobel.jpg"), sobel_new)
    cv2.imwrite(os.path.join(output_dir, "Gauss_New_Sobel.jpg"), gauss_sobel_full)

    print("Обработка завершена.")


if __name__ == "__main__":
    main()

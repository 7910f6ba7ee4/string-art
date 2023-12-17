import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

img = Image.open("../input/happy-smiling-man-giving-thumbs-676406-1378160142.jpg")
img = img.convert("L")

data = np.asarray(img)
radius = math.hypot(data.shape[0], data.shape[1])


def fourier_masker_vert(image, mask_val, c_avoid=20, width=4, f_size=15):
    im = np.fft.fftshift(np.fft.fft2(image))
    center_stop = int(im.shape[0] // 2 - c_avoid)
    width_left = (im.shape[1] - width) // 2
    width_right = (im.shape[1] + width) // 2
    im[:center_stop, width_left:width_right] = mask_val
    im[-center_stop:, width_left:width_right] = mask_val
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(np.log(abs(im)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize=f_size)
    ax[1].imshow(image, cmap='gray')
    ax[1].set_title('Greyscale Image', fontsize=f_size)
    ax[2].imshow(abs(np.fft.ifft2(im)),
                 cmap='gray')
    ax[2].set_title('Transformed Greyscale Image',
                    fontsize=f_size)


def fourier_masker_horz(image, mask_val, c_avoid=20, width=4, f_size=15):
    im = np.fft.fftshift(np.fft.fft2(image))
    center_stop = int(im.shape[1] // 2 - c_avoid)
    width_left = (im.shape[0] - width) // 2
    width_right = (im.shape[0] + width) // 2
    im[width_left:width_right, :center_stop] = mask_val
    im[width_left:width_right, -center_stop:] = mask_val
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(np.log(abs(im)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize=f_size)
    ax[1].imshow(image, cmap='gray')
    ax[1].set_title('Greyscale Image', fontsize=f_size)
    ax[2].imshow(abs(np.fft.ifft2(im)),
                 cmap='gray')
    ax[2].set_title('Transformed Greyscale Image',
                    fontsize=f_size)


fourier_masker_horz(data, 1, width=40)
fourier_masker_vert(data, 1, width=40)

# fft_data = np.fft.fft2(data)
# plt.imshow(np.log(abs(np.fft.fftshift(fft_data))), cmap="gray")
plt.show()

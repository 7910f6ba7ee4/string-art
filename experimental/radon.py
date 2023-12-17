import math

import radontea
from numpy.fft import fft
from skimage.transform import iradon, radon, iradon_sart
from radontea import fan
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


img = Image.open("../input/semi.jpg")
img = img.convert("L")
img = np.asarray(img)
img = np.pad(img, 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(img, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(img.shape) // 8, endpoint=False)
print(theta.shape, max(img.shape) // 8)
sinogram = radon(img, theta=theta)
print(sinogram.shape)
print(sinogram)
print(fft(sinogram, axis=0))
dx, dy = 0.5 * 180.0 / max(img.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()

reconstruction_fbp = iradon(sinogram, theta=theta)
# error = reconstruction_fbp - img
# print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
# ax2.set_title("Reconstruction error\nFiltered back projection")
# ax2.imshow(reconstruction_fbp - img, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()

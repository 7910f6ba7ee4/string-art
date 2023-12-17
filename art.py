import math

from skimage.transform import iradon, radon
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def resize_image(image: Image, length: int) -> Image:
    """
    Resize an image to a square. Can make an image bigger to make it fit or smaller if it doesn't fit. It also crops
    part of the image.

    credit: https://stackoverflow.com/questions/43512615/reshaping-rectangular-image-to-square

    :param self:
    :param image: Image to resize.
    :param length: Width and height of the output image.
    :return: Return the resized image.
    """

    """
    Resizing strategy : 
     1) We resize the smallest side to the desired dimension (e.g. 1080)
     2) We crop the other side so as to make it fit with the same length as the smallest side (e.g. 1080)
    """
    if image.size[0] < image.size[1]:
        # The image is in portrait mode. Height is bigger than width.

        # This makes the width fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((length, int(image.size[1] * (length / image.size[0]))))

        # Amount of pixel to lose in total on the height of the image.
        required_loss = (resized_image.size[1] - length)

        # Crop the height of the image so as to keep the center part.
        resized_image = resized_image.crop(
            box=(0, required_loss / 2, length, resized_image.size[1] - required_loss / 2))

        # We now have a length*length pixels image.
        return resized_image
    else:
        # This image is in landscape mode or already squared. The width is bigger than the heihgt.

        # This makes the height fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((int(image.size[0] * (length / image.size[1])), length))

        # Amount of pixel to lose in total on the width of the image.
        required_loss = resized_image.size[0] - length

        # Crop the width of the image so as to keep 1080 pixels of the center part.
        resized_image = resized_image.crop(
            box=(required_loss / 2, 0, resized_image.size[0] - required_loss / 2, length))

        # We now have a length*length pixels image.
        return resized_image


class Spoke:
    def __init__(self, num, n_spokes, radius):
        self.x = self._get_spoke_x(num, n_spokes, radius)
        self.y = self._get_spoke_y(num, n_spokes, radius)
        self.rad = num / n_spokes * 2 * math.pi
        self.unvisited = [x for x in range(1, n_spokes + 1) if x != num]

    @staticmethod
    def _get_spoke_x(num, n_spokes, radius):
        return radius + radius * 0.999999 * np.cos(num * 2 * math.pi / n_spokes)

    @staticmethod
    def _get_spoke_y(num, n_spokes, radius):
        return radius + radius * 0.999999 * np.sin(num * 2 * math.pi / n_spokes)


class ComputeStringArt:
    def __init__(self, file, n_spokes, radon_res, repetitions):
        image = Image.open(file).convert("L")
        image = resize_image(image, max(image.size))
        self.image = np.asarray(image)
        self.radius = self.image.shape[0] // 2
        self.string_image = np.zeros([self.radius * 2 + 1, self.radius * 2 + 1], dtype=np.float64)
        self.theta = np.linspace(0., 180., radon_res, endpoint=False)
        self.radon_transform = radon(self.image, theta=self.theta)
        self.working_radon = self.radon_transform.copy()
        self.n_spokes = n_spokes

        self.spoke_map = {num: Spoke(num, self.n_spokes, self.radius) for num in range(1, self.n_spokes + 1)}
        self.spokes_per_rad = self.n_spokes / (2 * math.pi)

        self.spoke_order = []
        # first_spokes = self._find_first_spokes()
        self.spoke_order.append(self._find_first_spokes())
        # print(self.spoke_map[first_spokes[0]].rad, math.radians(first_spokes[0]))
        # self.spoke_order.append(first_spokes[1])
        # print(f"spoke states: {self.spoke_order}")
        working_spoke = self.spoke_order[-1][1]
        for i in range(repetitions):
            new_spoke = self._find_best_spoke(working_spoke)
            # new_spokes = self._find_first_spokes()
            self.spoke_order.append((working_spoke, new_spoke))
            working_spoke = new_spoke
            # print(f"spoke states: {self.spoke_order}")

    def _find_project_pos(self, point_rad, proj_rad):
        return self.radius * math.cos(proj_rad - point_rad)

    def _find_opposite_spoke_search(self, spoke):
        return np.array([x for x in self.theta if abs(self.spoke_map[spoke].rad - x - math.pi / 2) > math.pi / 90])

    def _find_best_spoke(self, spoke):
        best_spoke = None
        best_spoke_val = None

        for rad_idx, degree in enumerate(self._find_opposite_spoke_search(spoke)):
            rad = math.radians(degree)
            pos = self._find_project_pos(self.spoke_map[spoke].rad, rad)
            val = self.radon_transform[int(pos), rad_idx]
            if best_spoke_val is None or val > best_spoke_val:
                s1, s2 = self._coords_to_spokes(pos, rad)
                # print(f"current spoke rad: {self.spoke_map[spoke].rad}, plane rad: {rad}")
                # print(f"position: {pos} out of radius: {self.radius}")
                # print(f"spoke 1: {s1}, spoke 2: {s2}, original spoke: {spoke}")
                if s1 - spoke == 0 or (s1 == 0 and spoke == self.n_spokes):
                    consider_spoke = s2
                elif s2 - spoke == 0 or (s2 == 0 and spoke == self.n_spokes):
                    consider_spoke = s1
                else:
                    consider_spoke = None
                    raise AssertionError()
                if consider_spoke in self.spoke_map[spoke].unvisited:
                    best_spoke = consider_spoke
                    best_spoke_val = val
                    # print(f"VAAAALLL: {val}")

        assert best_spoke_val is not None, "all spokes full"
        self.spoke_map[spoke].unvisited.remove(best_spoke)
        self.spoke_map[best_spoke].unvisited.remove(spoke)
        return best_spoke

    def _coords_to_spokes(self, proj_pos, proj_rad):
        spoke1 = self.spokes_per_rad * (math.pi / 2 - math.asin(proj_pos / self.radius) + proj_rad)
        spoke2 = self.spokes_per_rad * (-math.pi / 2 + math.asin(proj_pos / self.radius) + proj_rad)
        # print(spoke1, spoke2)
        spoke1 = round((spoke1 + self.n_spokes) % self.n_spokes)
        spoke2 = round((spoke2 + self.n_spokes) % self.n_spokes)
        return spoke1, spoke2

    def _find_first_spokes(self):
        proj_pos, proj_deg = np.unravel_index(np.argmin(self.working_radon[2:, 1:], axis=None),
                                              self.working_radon[2:, 1:].shape)
        # print(proj_pos, self.radius)
        # print(self.working_radon[proj_pos, proj_deg])
        # print(self.working_radon[proj_pos, proj_deg])
        spoke1, spoke2 = self._coords_to_spokes(proj_pos - self.radius, math.radians(self.theta[proj_deg]))
        self.working_radon[proj_pos + 2, proj_deg + 1] = 1000000000
        # print("spok", spoke1, spoke2)
        # print(proj_pos, proj_deg, self.working_radon[proj_pos][proj_deg], self.working_radon.shape)
        # print(self.working_radon[65:75, 91])
        # print(self.radon_transform[65:75, 85:95])
        spoke1 = self.n_spokes if spoke1 == 0 else spoke1
        spoke2 = self.n_spokes if spoke2 == 0 else spoke2
        if spoke2 in self.spoke_map[spoke1].unvisited:
            self.spoke_map[spoke1].unvisited.remove(spoke2)
        if spoke1 in self.spoke_map[spoke2].unvisited:
            self.spoke_map[spoke2].unvisited.remove(spoke1)
        return spoke1, spoke2

    def get_spoke_order(self):
        return self.spoke_order.copy()

    def _fpart(self, x):
        return x - math.floor(x)

    def _rfpart(self, x):
        return 1 - self._fpart(x)

    def _plot(self, x, y, c):
        self.string_image[y][x] = max(c, self.string_image[y][x])

    def _draw_line(self, x0, y0, x1, y1):
        steep = abs(y1 - y0) > abs(x1 - x0)

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0

        if dx == 0.0:
            gradient = 1.0
        else:
            gradient = dy / dx

        xend = round(x0)
        yend = y0 + gradient * (xend - x0)
        xgap = self._rfpart(x0 + 0.5)
        xpxl1 = xend
        ypxl1 = math.floor(yend)
        if steep:
            self._plot(ypxl1, xpxl1, self._rfpart(yend) * xgap)
            self._plot(ypxl1 + 1, xpxl1, self._fpart(yend) * xgap)
        else:
            self._plot(xpxl1, ypxl1, self._rfpart(yend) * xgap)
            self._plot(xpxl1, ypxl1 + 1, self._fpart(yend) * xgap)
        intery = yend + gradient

        xend = round(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = self._fpart(x1 + 0.5)
        xpxl2 = xend
        ypxl2 = math.floor(yend)
        if steep:
            self._plot(ypxl2, xpxl2, self._rfpart(yend) * xgap)
            self._plot(ypxl2 + 1, xpxl2, self._fpart(yend) * xgap)
        else:
            self._plot(xpxl2, ypxl2, self._rfpart(yend) * xgap)
            self._plot(xpxl2, ypxl2 + 1, self._fpart(yend) * xgap)

        if steep:
            for x in range(xpxl1 + 1, xpxl2):
                self._plot(math.floor(intery), x, self._rfpart(intery))
                self._plot(math.floor(intery) + 1, x, self._fpart(intery))
                intery = intery + gradient
        else:
            for x in range(xpxl1 + 1, xpxl2):
                self._plot(x, math.floor(intery), self._rfpart(intery))
                self._plot(x, math.floor(intery) + 1, self._fpart(intery))
                intery = intery + gradient

    def render(self, verbose):
        self.string_image = np.zeros([self.radius * 2 + 2, self.radius * 2 + 2], dtype=np.float64)
        for start, end in self.spoke_order:
            if verbose:
                print(f"Drawing line from ("
                      f"{self.spoke_map[start].x}, "
                      f"{self.spoke_map[start].y}) to "
                      f"({self.spoke_map[end].x},"
                      f"{self.spoke_map[end].y})")
            self._draw_line(
                self.spoke_map[start].x,
                self.spoke_map[start].y,
                self.spoke_map[end].x,
                self.spoke_map[end].y)

    def display_summary(self, file):
        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()
        ax[0].imshow(1 - self.string_image, cmap='grey')
        ax[0].set_title("String Art")
        ax[1].imshow(self.image, cmap='grey')
        ax[1].set_title("Original Image")
        ax[2].imshow(self.radon_transform, cmap='grey')
        ax[2].set_title("Radon Transform")
        ax[2].set_xlabel("Projection Degree")
        ax[2].set_ylabel("Projection Position")
        ax[3].imshow(iradon(self.radon_transform), cmap='grey')
        ax[3].set_title("Inverse Radon Recreation")
        fig.tight_layout()
        fig.savefig(file)

    def display_art(self, file):
        fig, ax = plt.subplots(1)
        ax.imshow(1 - self.string_image, cmap='grey')
        fig.savefig(file)


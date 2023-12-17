import torch
import matplotlib.pyplot as plt
import math
import numpy as np


def fpart(x):
    return x - math.floor(x)


def rfpart(x):
    return 1 - fpart(x)


def swap(a, b):
    c = a
    a = b
    b = c


def plot(x, y, c, img):
    img[y][x] = max(c, img[y][x])


def drawLine(x0, y0, x1, y1, img):
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
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = math.floor(yend)
    if steep:
        plot(ypxl1, xpxl1, rfpart(yend) * xgap, img)
        plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap, img)
    else:
        plot(xpxl1, ypxl1, rfpart(yend) * xgap, img)
        plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap, img)
    intery = yend + gradient

    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = math.floor(yend)
    if steep:
        plot(ypxl2, xpxl2, rfpart(yend) * xgap, img)
        plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap, img)
    else:
        plot(xpxl2, ypxl2, rfpart(yend) * xgap, img)
        plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap, img)

    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            plot(math.floor(intery), x, rfpart(intery), img)
            plot(math.floor(intery) + 1, x, fpart(intery), img)
            intery = intery + gradient
    else:
        for x in range(xpxl1 + 1, xpxl2):
            plot(x, math.floor(intery), rfpart(intery), img)
            plot(x, math.floor(intery) + 1, fpart(intery), img)
            intery = intery + gradient


def render(lines, radius, n_spokes):
    img = np.zeros([radius * 2 + 2, radius * 2 + 2], dtype=np.float64)
    getx = lambda num: radius * 0.999999 * np.cos(num * 2 * math.pi / n_spokes) + radius
    gety = lambda num: radius - radius * 0.999999 * np.sin(num * 2 * math.pi / n_spokes)
    spokes = {num: (getx(num), gety(num)) for num in range(1, n_spokes + 1)}
    print(spokes)
    print(radius * 0.999999 * np.cos(270 * 2 * math.pi / n_spokes) + radius)
    for line in lines:
        print(f"{line}: {spokes[line]}")
    for line_idx in range(1, len(lines)):
        print(f"Drawing line from ("
              f"{spokes[lines[line_idx - 1]][0]}, "
              f"{spokes[lines[line_idx - 1]][1]}) to "
              f"({spokes[lines[line_idx]][0]},"
              f"{spokes[lines[line_idx]][1]})")
        drawLine(
            spokes[lines[line_idx - 1]][0],
            spokes[lines[line_idx - 1]][1],
            spokes[lines[line_idx]][0],
            spokes[lines[line_idx]][1],
            img)
    return 1 - img


lines = [1, 270, 130]
img = render([1, 270, 130], 100, 360)
plt.imshow(img, 'grey')
plt.savefig("atest.png")

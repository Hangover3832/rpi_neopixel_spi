# From https://www.rapidtables.com/convert/color/

import numpy as np
np.seterr(invalid='ignore')

def rgb_to_hsv(rgb: np.ndarray | list | tuple) -> np.ndarray:
    """
    Convert an RGB numpy array to an HSV numpy array.

    :param rgb: Input array of shape (..., 3) with RGB values in range [0, 1].
    :type rgb: numpy.ndarray

    :return: Output array of shape (..., 3) with HSV values in ranges:
        H: [0, 1], S: [0, 1], V: [0, 1]
    :rtype: numpy.ndarray
    """

    # Ensure the input is a numpy array and in the correct shape
    rgb = np.asarray(rgb, dtype=np.float32)[..., :3]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    s = np.where(maxc == 0.0, 0.0, (maxc - minc) / maxc)

    # Compute Hue (H)
    rc = np.where(maxc == minc, 0.0, (maxc - r) / (maxc - minc))
    gc = np.where(maxc == minc, 0.0, (maxc - g) / (maxc - minc))
    bc = np.where(maxc == minc, 0.0, (maxc - b) / (maxc - minc))

    h = np.where(maxc == minc, 0,
                 np.where(r == maxc, bc - gc,
                 np.where(g == maxc, 2.0 + rc - bc,
                 4.0 + gc - rc)))

    h = (h / 6.0) % 1.0

    # Stack HSV components
    hsv = np.stack([h, s, v], axis=-1)

    return hsv


def hsv_to_rgb(hsv: np.ndarray | tuple | list) -> np.ndarray:

    hsv = np.asarray(hsv, dtype=np.float32)[..., :3]
    h, s, v = 360 * hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s

    # x = c * (1 - |(h/60) % 2 - 1|)
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = v - c

    # prepare arrays for r', g', b'
    rp = np.zeros_like(h)
    gp = np.zeros_like(h)
    bp = np.zeros_like(h)

    # create masks for each region
    mask0 = (h >= 0) & (h < 60)
    mask1 = (h >= 60) & (h < 120)
    mask2 = (h >= 120) & (h < 180)
    mask3 = (h >= 180) & (h < 240)
    mask4 = (h >= 240) & (h < 300)
    mask5 = (h >= 300) & (h < 360)

    rp[mask0] = c[mask0]; gp[mask0] = x[mask0]; bp[mask0] = 0.0
    rp[mask1] = x[mask1]; gp[mask1] = c[mask1]; bp[mask1] = 0.0
    rp[mask2] = 0.0;      gp[mask2] = c[mask2]; bp[mask2] = x[mask2]
    rp[mask3] = 0.0;      gp[mask3] = x[mask3]; bp[mask3] = c[mask3]
    rp[mask4] = x[mask4]; gp[mask4] = 0.0;      bp[mask4] = c[mask4]
    rp[mask5] = c[mask5]; gp[mask5] = 0.0;      bp[mask5] = x[mask5]

    # add m to match value and stack
    r = rp + m
    g = gp + m
    b = bp + m

    result = np.stack([r, g, b], axis=-1)
    return result


def run_test():
    from colors import SOME_COLORS

    print(hsv := rgb_to_hsv(list(SOME_COLORS['other'])))
    print(hsv_to_rgb(hsv))

    #testing on single color vectors:
    for i, (name, color) in enumerate(SOME_COLORS.items()):
        hsv = rgb_to_hsv(color)
        rgb = hsv_to_rgb(hsv)
        print(f"[{i:2}] {name:<10} {np.round(color, 3)}")
        print(f"[{i:2}] {'HSV':<10} {np.round(hsv, 3)}")
        print(f"[{i:2}] {'RGB':<10} {np.round(rgb, 3)}")
        print()

    #testing on color buffer array:
    buffer = np.vstack(list(SOME_COLORS.values()))
    print(np.round(buffer, 3))
    buffer = rgb_to_hsv(buffer)
    print(np.round(buffer, 3))
    buffer = hsv_to_rgb(buffer)
    print(np.round(buffer, 3))

        

if __name__ == '__main__':
    run_test()
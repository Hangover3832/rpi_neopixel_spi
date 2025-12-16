from matplotlib.colors import to_rgb
import numpy as np
from numpy.polynomial import Polynomial as Poly
from enum import Enum, auto
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_yiq, yiq_to_rgb, rgb_to_hls, hls_to_rgb

class ColorMode(Enum):

    # All color mode conversions from_rgb and to_rgb are available in the colorsys module
    def from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.max()<=1.0 and rgb.min()>= 0.0
        return {
            ColorMode.RGB: rgb[0:3],
            ColorMode.HSV: np.array(rgb_to_hsv(*rgb[0:3])),
            ColorMode.YIQ: np.array(rgb_to_yiq(*rgb[0:3])),
            ColorMode.HLS: np.array(rgb_to_hls(*rgb[0:3]))
        }[self]

    def to_rgb(self, value:np.ndarray) -> np.ndarray:
        assert value.max()<=1.0 and value.min()>= 0.0
        return {
            ColorMode.RGB: value[0:3],
            ColorMode.HSV: np.array(hsv_to_rgb(*value[0:3])),
            ColorMode.YIQ: np.array(yiq_to_rgb(*value[0:3])),
            ColorMode.HLS: np.array(hls_to_rgb(*value[0:3]))
        }[self]
    
    # Define all the missing color mode conversion methods via rgb:
    def from_hsv(self, hsv:np.ndarray) -> np.ndarray:
        return hsv if self == ColorMode.HSV else self.from_rgb(np.array(hsv_to_rgb(*hsv[0:3])))

    def to_hsv(self, value:np.ndarray) -> np.ndarray:
        return value if self == ColorMode.HSV else np.array(rgb_to_hsv(*self.to_rgb(value)))

    def from_yiq(self, yiq:np.ndarray) -> np.ndarray:
        return yiq if self == ColorMode.YIQ else self.from_rgb(np.array(yiq_to_rgb(*yiq[0:3])))

    def to_yiq(self, value:np.ndarray) -> np.ndarray:
        return value if self == ColorMode.YIQ else np.array(rgb_to_yiq(*self.to_rgb(value)))

    def from_hls(self, hls:np.ndarray) -> np.ndarray:
        return hls if self == ColorMode.HLS else self.from_rgb(np.array(hls_to_rgb(*hls[0:3])))

    def to_hls(self, value:np.ndarray) -> np.ndarray:
        return value if self == ColorMode.HLS else np.array(rgb_to_hls(*self.to_rgb(value)))

    RGB = auto()
    HSV = auto()
    YIQ = auto()
    HLS = auto()


class PixelOrder(Enum):
    """Any combination of R, G, B and W is possible"""

    @property
    def num(self) -> int:
        return len(self.name)
    
    RGB = auto()
    GRB = auto()
    RGBW = auto()
    GRBW = auto()


"""
The gamma system uses a polynomial regression to map the input value to the output value.
https://en.wikipedia.org/wiki/Polynomial_regression
"""

CUSTOM_GAMMA = np.array([
    0.0,    # value for 0% brightness
    0.25,   # value for 25% brightness
    0.50,   # value for 50% brightness
    0.75,   # value for 75% brightness
    1.0     # value for 100% brightness
]) # insert more values in between if desired

LINEAR_GAMMA = np.array([0.0, 1.0])
INVERSE_GAMMA = np.array([1.0, 0.0])
SQUARE_GAMMA = np.array([0.0, 0.25, 1.0])

SIMPLE_GAMMA = np.array([0.0, 0.214, 1.0]) # sRGB 21.4% middle grey
"""This is quite close to the square gamma where the middle grey is 25%"""

DEFAULT_GAMMA = np.array([0.0, 0.11, 0.18, 0.35, 1.0]) # 18% middle grey + my own magic
""" 
Note that I created this masterpiece of gamma based on the 18% middle grey principle
(Munsell, Sloan & Godlove, see https://en.wikipedia.org/wiki/Middle_gray) and just by my own subjective
perception of brightness on a particular neopixel stripe.
"""

SRGB_GAMMA = np.array([0.0, 0.15, 0.214, 0.37, 1.0])
# like default gamma but shifted towards sRGB

NO_DARK_GAMMA = np.array([0.02, 1.0])
# no more dark pixels :-()

CRAZY_GAMMA = np.array([1.0, 0.0, 1.0]) 
# dark pixels are bright, bright pixels are bright and middle pixels are dark ;-)


def create_gamma_function(values_out: np.ndarray) -> Poly:
    """apply a (n-1)th degree polynomial fit using n(values_out) distinct data points.
    Args:
        values_out (np.ndarray): Array of output values for the polynomial fit.
    Returns:
        A polynomial function that maps input values in [0, 1] to the specified output values [0, 1].
    """
    n = len(values_out)
    assert n > 1, "Gamma polynomial must have more than 1 data points"
    vIn = np.linspace(0.0, 1.0, n) # create the linear input space from 0..1
    return Poly.fit(vIn, values_out, deg=n-1, domain=(0.0, 1.0), window=(0.0, 1.0))


default_gamma = create_gamma_function(DEFAULT_GAMMA)
srgb_gamma =    create_gamma_function(SRGB_GAMMA)
simple_gamma =  create_gamma_function(SIMPLE_GAMMA)
no_dark_gamma = create_gamma_function(NO_DARK_GAMMA)
crazy_gamma =   create_gamma_function(CRAZY_GAMMA)
square_gamma =  create_gamma_function(SQUARE_GAMMA)
linear_gamma =  create_gamma_function(LINEAR_GAMMA)
inverse_gamma = create_gamma_function(INVERSE_GAMMA)
custom_gamma =  create_gamma_function(CUSTOM_GAMMA)


def plot_gamma_functions() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes._axes import Axes
    from matplotlib.figure import Figure

    x = np.linspace(0.0, 1.0, 25)
    splt = plt.subplots()
    fig: Figure = splt[0]
    ax: Axes = splt[1]
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.set_title("Gamma Functions")
    ax.set_xlabel("Input Value")
    ax.set_ylabel("LED Brightness")

    ax.plot(x, default_gamma(x), linewidth=2.0, label='default_gamma')
    ax.plot(x, srgb_gamma(x), linewidth=2.0, label='srgb_gamma')
    ax.plot(x, simple_gamma(x), linewidth=2.0, label='simple_gamma')
    ax.plot(x, no_dark_gamma(x), linewidth=2.0, label='no_dark_gamma')
    ax.plot(x, crazy_gamma(x), linewidth=2.0, label='crazy_gamma')
    ax.plot(x, square_gamma(x), linewidth=2.0, label='square_gamma')
    ax.plot(x, linear_gamma(x), linewidth=2.0, label='linear_gamma')
    ax.plot(x, inverse_gamma(x), linewidth=2.0, label='inverse_gamma')
    ax.plot(x, custom_gamma(x), linewidth=2.0, label='custom_gamma')

    fig.legend()
    plt.show()

if __name__ == '__main__':
    plot_gamma_functions()

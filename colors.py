import numpy as np
from numpy.polynomial import Polynomial as Poly
from enum import Enum, auto


class ColorMode(Enum):
    RGB = auto()
    HSV = auto()
    YIQ = auto()
    HLS = auto()


class PixelOrder(Enum):
    """Any order of R, G, B and W is possible"""
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


def _poly_fit(values_out: np.ndarray) -> Poly:
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


custom_gamma =  _poly_fit(CUSTOM_GAMMA)
default_gamma = _poly_fit(DEFAULT_GAMMA)
srgb_gamma =    _poly_fit(SRGB_GAMMA)
simple_gamma =  _poly_fit(SIMPLE_GAMMA)
no_dark_gamma = _poly_fit(NO_DARK_GAMMA)
crazy_gamma =   _poly_fit(CRAZY_GAMMA)
square_gamma =  lambda x: np.square(x)
linear_gamma =  lambda x: x
inverse_gamma = lambda x: 1.0-x

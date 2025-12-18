from typing import Callable
from humanfriendly import coerce_pattern
import numpy as np
from numpy.polynomial import Polynomial as Poly
from enum import Enum, auto
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_yiq, yiq_to_rgb, rgb_to_hls, hls_to_rgb


class ColorMode(Enum):

    # All color mode conversions from_rgb and to_rgb are available in the colorsys module
    def from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.clip(rgb, 0.0, 1.0)
        # assert rgb.max()<=1.0 and rgb.min()>= 0.0
        return {
            ColorMode.RGB: rgb[0:3],
            ColorMode.HSV: np.array(rgb_to_hsv(*rgb[0:3])),
            ColorMode.YIQ: np.array(rgb_to_yiq(*rgb[0:3])),
            ColorMode.HLS: np.array(rgb_to_hls(*rgb[0:3]))
        }[self]

    def to_rgb(self, value:np.ndarray) -> np.ndarray:
        # assert value.max()<=1.0 and value.min()>= 0.0
        value = np.clip(value, 0.0, 1.0)
        return {
            ColorMode.RGB: value[0:3],
            ColorMode.HSV: np.array(hsv_to_rgb(*value[0:3])),
            ColorMode.YIQ: np.array(yiq_to_rgb(*value[0:3])),
            ColorMode.HLS: np.array(hls_to_rgb(*value[0:3]))
        }[self]
    
    # Define all the missing color mode conversion methods via rgb:
    def from_hsv(self, hsv:np.ndarray) -> np.ndarray:
        hsv = np.clip(hsv, 0.0, 1.0)
        return hsv if self == ColorMode.HSV else self.from_rgb(np.array(hsv_to_rgb(*hsv[:3])))

    def to_hsv(self, value:np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value if self == ColorMode.HSV else np.array(rgb_to_hsv(*self.to_rgb(value)))

    def from_yiq(self, yiq:np.ndarray) -> np.ndarray:
        yiq = np.clip(yiq, 0.0, 1.0)
        return yiq if self == ColorMode.YIQ else self.from_rgb(np.array(yiq_to_rgb(*yiq[:3])))

    def to_yiq(self, value:np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value if self == ColorMode.YIQ else np.array(rgb_to_yiq(*self.to_rgb(value)))

    def from_hls(self, hls:np.ndarray) -> np.ndarray:
        hls = np.clip(hls, 0.0, 1.0)
        return hls if self == ColorMode.HLS else self.from_rgb(np.array(hls_to_rgb(*hls[:3])))

    def to_hls(self, value:np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value if self == ColorMode.HLS else np.array(rgb_to_hls(*self.to_rgb(value)))

    @staticmethod
    def default() -> 'ColorMode':
        return ColorMode.HSV

    RGB = auto()
    HSV = auto()
    YIQ = auto()
    HLS = auto()


class PixelOrder(Enum):
    """Any combination of R, G, B and W is possible"""

    @property
    def num(self) -> int:
        """Return the number of LED per pixel (3 or 4)"""
        return len(self.name)

    @property
    def blank(self) -> np.ndarray:
        """Return color black value appropriate for the pixel type (RGB or RGBW)"""
        return np.array([0., 0., 0., 0.]) if self.num > 3 else np.array([0., 0., 0.])

    @classmethod
    def default(cls) -> 'PixelOrder':
        return cls.GRB

    RGB     = auto()
    GRB     = auto()
    RGBW    = auto()
    GRBW    = auto()


"""
The gamma system uses a polynomial regression to map the input value to the output value.
https://en.wikipedia.org/wiki/Polynomial_regression
"""

CUSTOM_GAMMA = np.array([
    0.0,    # value for 0% brightness
    0.25,   # value for 25% brightness
    0.5,   # value for 50% brightness
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


def _create_gamma_function(values_out: np.ndarray) -> np.poly1d:
    """Calculate a polynomial that fits the given output values **exactly**."""
    return np.poly1d(np.linalg.solve(np.vander(np.linspace(0.0, 1.0, n:=len(values_out)), N=n), values_out))


def create_gamma_function(values_out: np.ndarray) -> Poly:
    """Apply a (n-1)th degree polynomial least quare fit using n(values_out) distinct data points.

    :param values_out: Array of output values for the polynomial fit. For the input values, a linear space is calculated.
    :type values_out: np.ndarray
    :returns: A polynomial function that maps input values in [0, 1] to the specified output values [0, 1].
    :rtype: numpy.polynomial.Polynomial
    """
    assert (n := len(values_out)) > 1, "Gamma polynomial must have more than 1 data points"
    values_in = np.linspace(0.0, 1.0, n) # create the linear input space from 0..1
    return Poly.fit(values_in, values_out, deg=n-1, window=(0.0, 1.0), domain=(0.0, 1.0))


class G(Enum):
    """Gamma functions enum class."""

    @classmethod
    def Default(cls) -> Callable:
        return cls.default.value
    
    @classmethod
    def plot(cls, function: Callable | None = None, builtin:bool=True) -> None:
        """
        Gamma function plotter.
        
        :param function: The function to be plotted along the built-in functions.
        :type function: Callable | None
        """

        import matplotlib.pyplot as plt
        from matplotlib.axes._axes import Axes
        from matplotlib.figure import Figure

        x: np.ndarray = np.linspace(0.0, 1.0, 25)
        splt = plt.subplots(2)
        fig: Figure = splt[0]
        ax: Axes = splt[1]
        ax[0].set(xlim=(0.0, 1.0), ylim=(0.0, 1.0)) # type: ignore
        ax[1].set(xlim=(0.0, 1.0), ylim=(0.0, 1.0)) # type: ignore
        ax[0].set_title("Custom Gamma Function") # type: ignore
        ax[1].set_title("Built-in Gamma Functions") # type: ignore
        ax[1].set_xlabel("Input Value") # type: ignore
        ax[1].set_ylabel("LED Brightness") # type: ignore

        # Plot all bilt-in gamma functions
        if builtin:
            for f in G:
                ax[1].plot(x, f.value(x), linewidth=2.0, label=f.name) # type: ignore

        if function:
            # Plot a custom gamma function
            ax[0].plot(x, function(x), linewidth=2.0) # type: ignore

        ax[1].legend() # type: ignore
        plt.show()


    default = create_gamma_function(DEFAULT_GAMMA)
    srgb    = create_gamma_function(SRGB_GAMMA)
    simple  = create_gamma_function(SIMPLE_GAMMA)
    no_dark = create_gamma_function(NO_DARK_GAMMA)
    crazy   = create_gamma_function(CRAZY_GAMMA)
    square  = create_gamma_function(SQUARE_GAMMA)
    linear  = create_gamma_function(LINEAR_GAMMA)
    inverse = create_gamma_function(INVERSE_GAMMA)
    custom  = create_gamma_function(CUSTOM_GAMMA)


def main():
    test_gamma = lambda x: 0.5 + np.sin(x * 2 * np.pi) / 2
    G.plot(test_gamma)    


if __name__ == '__main__':
    main()

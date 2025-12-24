from typing import Callable, Tuple
import numpy as np
from numpy.polynomial import Polynomial as Poly
from enum import Enum, auto
# from colorsys import rgb_to_yiq, yiq_to_rgb, rgb_to_hls, hls_to_rgb
from color_conversions import rgb_to_hsv, hsv_to_rgb

SOME_COLORS = {
    'red':     np.array([1., 0., 0., 0.]),
    'green':   np.array([0., 1., 0., 0.]),
    'blue':    np.array([0. ,0., 1., 0.]),
    'yellow':  np.array([1., 1., 0., 0.]),
    'aqua':    np.array([0., 1., 1., 0.]),
    'pink':    np.array([1., 0., 1., 0.]),
    'white':   np.array([1., 1., 1., 0.]),
    'other':   np.array([0.7, 0.5, 0.2, 0.]),
}


def _create_gamma_function(values_out:np.ndarray) -> np.poly1d:
    """Calculate a polynomial that fits the given output values **exactly**."""
    return np.poly1d(np.linalg.solve(np.vander(np.linspace(0.0, 1.0, n:=len(values_out)), N=n), values_out))


def create_gamma_function(values_out:np.ndarray, values_in:np.ndarray | None = None) -> Poly:
    """Apply a (n-1)th degree polynomial least quare fit using n(values_out) distinct data points.

    :param values_out: Array of output values for the polynomial fit. For the input values, a linear space is calculated.
    :type values_out: np.ndarray
    :returns: A polynomial function that maps input values in [0, 1] to the specified output values [0, 1].
    :rtype: numpy.polynomial.Polynomial
    """
    assert (n := len(values_out)) > 1, "Gamma polynomial must have more than 1 data points"
    if values_in is None:
        values_in = np.linspace(0.0, 1.0, n) # create the linear input space from 0..1
    return Poly.fit(values_in, values_out, deg=n-1, window=(0.0, 1.0), domain=(0.0, 1.0))


class TempColor:
    """
    Black body temeratur spectrum.
    C = red yellow white blue
    T = 0  1/3  2/3   3/3   1"""
    R = (1.0,  1.0,  1.0, 0.0)
    G = (0.0,  1.0,  0.9, 0.0) # avoid the appearance of green in the spectrum
    B = (0.0,  0.0,  0.9, 1.0)
    RGB = np.array([R, G, B])

    kelvin_R = _create_gamma_function(np.array([1.0, 0.55, 0.333, 0.0]))
    kelvin_B = _create_gamma_function(np.array([0.0, 0.05, 0.333, 1.0]))
    kelvin_G = 1.0 - kelvin_R - kelvin_B


    @classmethod
    def _temperature_to_RGB(cls, temp:float) -> tuple[float, float, float]:
        return cls.kelvin_R(temp), cls.kelvin_G(temp), cls.kelvin_B(temp)

    @classmethod
    def temperature_to_RGB(cls, temp:float) -> tuple[float, float, float]:
        if temp <= 1.0/3:
            interval = np.array((0.0, 1.0/3))
            rgb = cls.RGB[:, :2]
        elif temp >= 2.0/3:
            interval = np.array((2.0/3, 1.0))
            rgb = cls.RGB[:, 2:]
        else:
            interval = np.array((1.0/3, 2.0/3))
            rgb = cls.RGB[:, 1:-1]

        r = np.interp(temp, interval, rgb[0])
        g = np.interp(temp, interval, rgb[1])
        b = np.interp(temp, interval, rgb[2])

        return r, g, b


class ColorMode(Enum):

    def kelvin_to_rgb(self, kelvin: float) -> np.ndarray:
        return np.array(TempColor.temperature_to_RGB(kelvin))

    # All color mode conversions from_rgb and to_rgb are available in the colorsys module
    def from_rgb(self, rgb: np.ndarray) -> np.ndarray:

        if self == ColorMode.RGB:
            return rgb[:3]  
        elif self == ColorMode.HSV:
            return rgb_to_hsv(rgb)
        else:
            raise NotImplementedError(f"Color mode {self.value} is currently not implemented.")


    def to_rgb(self, value:np.ndarray) -> np.ndarray:

        if self == ColorMode.RGB:
            return value[:3]  
        elif self == ColorMode.HSV:
            return hsv_to_rgb(value)
        else:
            raise NotImplementedError(f"Color mode {self.value} is currently not implemented.")


    # Define all the missing color mode conversion methods via rgb:
    def from_hsv(self, hsv:np.ndarray) -> np.ndarray:
        return hsv if self == ColorMode.HSV else self.from_rgb(hsv_to_rgb(hsv))

    def to_hsv(self, value:np.ndarray) -> np.ndarray:
        return value if self == ColorMode.HSV else rgb_to_hsv(self.to_rgb(value))

    """ 
    def from_yiq(self, yiq:np.ndarray) -> np.ndarray:
        return yiq if self == ColorMode.YIQ else self.from_rgb(np.array(yiq_to_rgb(*yiq[:3])))

    def to_yiq(self, value:np.ndarray) -> np.ndarray:
        return value if self == ColorMode.YIQ else np.array(rgb_to_yiq(*self.to_rgb(value[:3])))

    def from_hls(self, hls:np.ndarray) -> np.ndarray:
        return hls if self == ColorMode.HLS else self.from_rgb(np.array(hls_to_rgb(*hls[:3])))

    def to_hls(self, value:np.ndarray) -> np.ndarray:
        return value if self == ColorMode.HLS else np.array(rgb_to_hls(*self.to_rgb(value[:3])))
    """


    RGB = auto()
    HSV = auto()
    # YIQ = auto()
    # HLS = auto()


class PixelOrder(Enum):
    """Any combination of R, G, B and W is possible"""

    @property
    def num(self) -> int:
        """Return the number of LED per pixel (3 or 4)"""
        return len(self.name)

    @property
    def blank(self) -> np.ndarray:
        """Return black color value appropriate for the pixel type (RGB or RGBW)"""
        return np.array([0., 0., 0., 0.]) if self.num > 3 else np.array([0., 0., 0.])


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
I created this masterpiece of gamma based on the 18% middle grey principle
(Munsell, Sloan & Godlove, see https://en.wikipedia.org/wiki/Middle_gray) and just by my own subjective
perception of brightness on a particular neopixel stripe.
"""

SRGB_GAMMA = np.array([0.0, 0.15, 0.214, 0.37, 1.0])
# like default gamma but shifted towards sRGB

NO_DARK_GAMMA = np.array([0.02, 1.0])
# no more dark pixels :-()

CRAZY_GAMMA = np.array([1.0, 0.0, 1.0]) 
# dark pixels are bright, bright pixels are bright and middle pixels are dark ;-)


class G(Enum):
    """Gamma functions enum class."""

    @classmethod
    def plot(cls, functions: list[Callable] | None = None) -> None:
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
        ax: Axes = splt[1] # type: ignore
        ax[1].set(xlim=(0.0, 1.0), ylim=(0.0, 1.0)) # type: ignore
        ax[0].set(xlim=(0.0, 1.0), ylim=(-0.1, 1.1)) # type: ignore
        ax[0].set_title("Custom Gamma Function") # type: ignore
        ax[1].set_title("Built-in Gamma Functions") # type: ignore
        ax[1].set_xlabel("Input Value") # type: ignore
        ax[1].set_ylabel("LED Brightness") # type: ignore

        # Plot all bilt-in gamma functions
        for f in G:
            ax[1].plot(x, f.value(x), linewidth=2.0, label=f.name) # type: ignore

        if functions:
            for f in functions:
                # Plot a custom gamma function
                ax[0].plot(x, f(x), linewidth=2.0) # type: ignore

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
    test_gamma = lambda x: .5 + np.sin(x * 2 * np.pi) / 2
    kelvin = TempColor.kelvin_R + TempColor.kelvin_G + TempColor.kelvin_B
    G.plot([TempColor.kelvin_R, TempColor.kelvin_G, TempColor.kelvin_B, kelvin])


if __name__ == '__main__':
    main()

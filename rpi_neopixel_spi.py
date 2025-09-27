"""
Raspberry Pi NeoPixel driver using SPI interface.
Author: AlexL
License: MIT
Github: https://github.com/Hangover3832/rpi_neopixel_spi
"""

from dataclasses import dataclass
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_yiq, yiq_to_rgb, rgb_to_hls, hls_to_rgb
from typing import Union, Callable
from numpy.typing import NDArray
from spidev import SpiDev
from threading import Lock


def gamma_square(value: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Applies a square gamma correction to the input value.
    Parameters:
        value (np.array or float): The input array of color values or a single float value.
    Returns:
        np.array or float: The gamma-corrected array or float value.
    """
    return np.clip(np.square(value), 0.0, 1.0)


def gamma_linear(value:np.ndarray) -> np.ndarray:
    """
    Applies a linear gamma correction (no change) to the input value.

    Parameters:
        value (np.array): The input array of color values.

    Returns:
        np.array: The same input array, unchanged.
    """
    return np.clip(value, 0.0, 1.0)


def gamma4g(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    4th order polynomial gamma correction function.
    Parameters:
        x (np.array or float): The input array of color values or a single float value.
    Returns:
        np.array or float: The gamma-corrected array or float value.
    """
    p1 = 0.11
    p2 = 0.35
    d = 0.5
    a = -16*d/3 + 128*p1/3 - 128*p2/9 + 16/3
    b = 32*d/3 - 224*p1/3 + 160*p2/9 - 16/3
    c = -19*d/3 + 32*p1 - 32*p2/9 + 1
    e = 0.
    return np.polynomial.Polynomial((e, d, c, b, a), domain=(0., 1.), window=(0., 1.))(x)


class RpiNeoPixelSPI:
    """    
    Driver for NeoPixel LEDs using SPI interface on a Raspberry PI.
    Args:
        num_pixels (int): Number of NeoPixel LEDs in the strip.
        device (int, optional): SPI device number. Defaults to 0.
            device=0 -> /dev/spidev0.0 uses BCM pins 8 (CE0), 9 (MISO), 10 (MOSI), 11 (SCLK)
            device=1 -> /dev/spidev0.1 uses BCM pins 7 (CE1), 9 (MISO), 10 (MOSI), 11 (SCLK)
        gamma_func (Callable, optional): Function to apply gamma correction. Defaults to gamma4g.
        color_mode (str, optional): Color mode for input values. Options are "RGB", "HSV", "YIQ", "HLS". Defaults to "HSV".
        brightness (float, optional): Brightness level (0.0 to 1.0). Defaults to 1.0.
        auto_write (bool, optional): If True, updates the LEDs automatically on value change. Defaults to False.
        pixel_order (str, optional): Order of color channels in the NeoPixel (e.g., "GRB", "RGBW"). Defaults to "GRB".
        clock_rate (int, optional): SPI clock rate in Hz. Defaults to CLOCK_800KHZ.
    Clock Rates:
        CLOCK_400KHZ  (1,625,000 Hz): Standard rate for WS2812 pixels
        CLOCK_800KHZ  (3,250,000 Hz): High-speed rate for WS2812B pixels
        CLOCK_1200KHZ (6,500,000 Hz): Maximum rate for some pixels
    """

    COLOR_RGB_BLACK     = np.array([0., 0., 0.])
    COLOR_RGB_BLACK_W   = np.array([0., 0., 0., 0.])
    COLOR_RGB_WHITE     = np.array([1., 1., 1.])
    COLOR_RGB_WHITE_W   = np.array([0., 0., 0., 1.])
    COLOR_MODES         = ["RGB", "HSV", "YIQ", "HLS"]
    PIXEL_ORDERS        = ["RGB", "RGBW", "GRB", "GRBW"]
    CLOCK_400KHZ        = 1_625_000
    CLOCK_800KHZ        = 3_250_000
    CLOCK_1200KHZ       = 6_500_000
    # Add constants for SPI encoding
    SPI_HIGH_BIT        = 0xC0
    SPI_LOW_BIT         = 0x80
    SPI_HIGH_BIT2       = 0x0C
    SPI_LOW_BIT2        = 0x08


    def __init__(self,
                num_pixels: int,
                *,
                device: int = 0,
                gamma_func: Callable | None = gamma4g,
                color_mode: str = "HSV", 
                brightness: float = 1.0, 
                auto_write: bool= False,
                pixel_order: str = "GRB",
                clock_rate: int = CLOCK_800KHZ
                ) -> None:

        self.__pixel_order = pixel_order.upper()
        if self.__pixel_order in self.PIXEL_ORDERS:
            pass
        else:
            raise ValueError(f"Unexpected pixel order: {self.__pixel_order}")

        self.__color_mode = color_mode.upper()
        if not self.__color_mode in self.COLOR_MODES:
            raise ValueError(f"Unexpected color mode: '{self.__color_mode}'")

        # Initialize SPI device
        if device not in [0, 1]:
            raise ValueError("Error: SPI device must be 0 or 1.")
        try:
            self._spi = SpiDev()
            self._spi.open(bus=0, device=device)
            self._spi.max_speed_hz = clock_rate
            self._spi.mode = 0
            self._spi.bits_per_word = 8
        except:
            raise RuntimeError("Error: Could not open SPI device. Ensure SPI is enabled in raspi-config and the device number is correct.")

        self.__num_pixels = num_pixels
        self.__brightness = float(np.clip(brightness, 0.0, 1.0))
        if gamma_func is None:
            self.__gamma_func = lambda x: x
        else:
            self.__gamma_func = gamma_func
        self.__auto_write = auto_write
        self.__pixel_buffer = np.zeros((self.__num_pixels, len(self.__pixel_order)), dtype=float)

         
        if len(self.__pixel_order) == 4:
            self.bits_per_pixel = 16
            self.__msb_mask = 0x80000000
            self.__c_mask = 0xFFFFFFFF
        else:
            self.bits_per_pixel = 12
            self.__msb_mask = 0x800000
            self.__c_mask = 0xFFFFFF

        # Pre-allocate buffer
        self.__spi_buffer = np.zeros([self.bits_per_pixel, self.num_pixels], dtype=np.uint8)


    def __to_RGB(self, value: np.ndarray, color_mode: str | None = None) -> np.ndarray:
        """Convert value from color_mode (or, if not provided, the current color mode) to RGB"""
        if color_mode is None:
            color_mode= self.__color_mode

        match color_mode.upper():
            case "RGB":
                result = value[0:3]
            case "HSV":
                result = np.array(hsv_to_rgb(*value[0:3]))
            case "YIQ":
                result = np.array(yiq_to_rgb(*value[0:3]))
            case "HLS":
                result = np.array(hls_to_rgb(*value[0:3]))
            case _:
                raise ValueError(f"Unexpected color mode: '{color_mode}'")

        if value.shape[0] > 3 and self._has_W:
            return np.append(result, value[3])
        else: 
            return result


    def __to_HSV(self, value: np.ndarray, color_mode: str | None ) -> np.ndarray:
        """Convert value from color_mode (or, if not provided, the current color mode) to HSV"""
        if color_mode is None:
            color_mode = self.__color_mode

        if color_mode.upper() == "HSV":
            return value
        else:
            rgb = self.__to_RGB(value, color_mode)
            hsv = np.array(rgb_to_hsv(*rgb[0:3]))

            if value.shape[0] > 3 and self._has_W:
                return np.append(hsv, value[3])
            else:
                return hsv
 

    def __from_RGB(self, rgb: np.ndarray, color_mode: str | None = None) -> np.ndarray:
        """Convert value from RGB to color_mode (or, if not provided, to the current color mode)"""
        if color_mode is None:
            color_mode =self.__color_mode

        match color_mode.upper():
            case "RGB":
                result = rgb[0:3]
            case "HSV":
                result = np.array(rgb_to_hsv(*rgb[0:3]))
            case "YIQ":
                result = np.array(rgb_to_yiq(*rgb[0:3]))
            case "HLS":
                result = np.array(rgb_to_hls(*rgb[0:3]))
            case _:
                raise ValueError(f"Unexpected color mode: '{color_mode}'")
        
        if rgb.shape[0] > 3 and self._has_W:
            return np.append(result, rgb[3])
        else: 
            return result
        

    def _write_buffer(self, rgb_buffer: np.ndarray | None = None) -> None: # type: ignore
        """
        Write pixel data to NeoPixels using SPI protocol.
        
        Args:
            buffer: Optional numpy array containing pixel data. 
                   If None, uses internal pixel buffer.
        """
        if rgb_buffer is None:
            rgb_buffer: np.ndarray = self.__pixel_buffer.copy()

        # Apply brightness and gamma correction, scale to [0, 255], and convert to uint8
        rgb_buffer = np.clip(np.round(self.__gamma_func(rgb_buffer * self.__brightness) * 255), 0, 255).astype(np.uint8)

        # rows are now pixels, columns are R,G,B,(W)
        # lets rearange the rgb_buffer to the correct pixel order
        # Here, we allow every possible pixel order with R,G,B and optional W
        rgb_buffer = rgb_buffer[:, [self.__pixel_order.index(c) for c in 'RGBW' if c in self.__pixel_order]]

        # Convert [r, g, b, (w)] uint8 to a single uint32:
        if self._has_W:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00_00, 0x1_00_00, 0x1_00, 1], dtype=np.uint32)
        else:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00, 0x1_00, 1], dtype=np.uint32)
        rgb_buffer = np.bitwise_or.reduce(rgb_buffer, axis=1, dtype=np.uint32)

        # shift out 2 bits of each pixel and encode to a byte for SPI transmission 
        for i in range(self.bits_per_pixel):
            bit1 = (rgb_buffer & self.__msb_mask).astype(bool)
            rgb_buffer = ((rgb_buffer << 1) & self.__c_mask).astype(np.uint32)
            bit2 = (rgb_buffer & self.__msb_mask).astype(bool) 
            rgb_buffer = ((rgb_buffer << 1) & self.__c_mask).astype(np.uint32)
            # encode 2 pixel bits into 1 SPI byte
            self.__spi_buffer[i] = (np.where(bit1, self.SPI_HIGH_BIT, self.SPI_LOW_BIT) | np.where(bit2, self.SPI_HIGH_BIT2, self.SPI_LOW_BIT2)).astype(np.uint8)

        # Send data to device
        self._spi.writebytes2(self.__spi_buffer.T.flatten()) # type: ignore


    def __setitem__(self, index: int, value: Union[np.ndarray, list, tuple], color_mode: str | None = None) -> None:
        rgb = self.__to_RGB(np.clip(value, 0.0, 1.0), color_mode=color_mode)

        if rgb.shape[0] == 3 and self._has_W:
            # rgb = np.append(rgb, self.__pixel_buffer[index][3])
            self.__pixel_buffer[index][0:3] = rgb
        else:
            self.__pixel_buffer[index] = rgb

        if self.__auto_write:
            self.show()


    def set_value(self, index: int | list[int], *values, color_mode: str | None = None) -> None:
        if isinstance(index, int):
            index = [index]

        for i in index:
            if isinstance(values[0], (list, np.ndarray, tuple)):
                self.__setitem__(i, values[0], color_mode=color_mode)
            else:
                self.__setitem__(i, values, color_mode=color_mode)


    def __getitem__(self, index: int) -> np.ndarray:
        return self.__from_RGB(self.__pixel_buffer[index])


    def fill(self, value: Union[np.ndarray, list, tuple], color_mode: str | None = None) -> None:
        rgb = self.__to_RGB(np.clip(value, 0.0, 1.0), color_mode=color_mode)
        self.__pixel_buffer [:] = rgb
        if self.__auto_write:
            self.show()


    def clear(self) -> None:
        """Clear all pixels by setting them to black."""
        black = self.COLOR_RGB_BLACK_W if self._has_W else self.COLOR_RGB_BLACK
        self.fill(black, color_mode="RGB")


    def show(self) -> None:
        self._write_buffer()


    def __call__(self, *args):
        # immediate update
        match len(args):
            case 0:
                pass
            case 1: # clear pixel at index
                if self._has_W:
                    self[args[0]] = RpiNeoPixelSPI.COLOR_RGB_BLACK_W
                else:
                    self[args[0]] = RpiNeoPixelSPI.COLOR_RGB_BLACK
            case 2: # set pixel at index(es)
                if isinstance(args[0], (list, tuple)):
                    for i in args[0]:
                        self[i] = args[1] 
                else:
                    self[args[0]] = args[1]
            case _:
                raise ValueError("Too many arguments!")
            
        if not self.__auto_write:
            self.show()


    @property
    def _has_W(self) -> bool:
        return self.__pixel_buffer.shape[1] > 3


    @property
    def gamma(self):
        return self.__gamma_func

    @gamma.setter
    def gamma(self, new_gamma) -> None:
        self.__gamma_func = new_gamma
        if self.__auto_write:
            self.show()


    @property 
    def color_mode(self) -> str:
        return self.__color_mode

    @color_mode.setter
    def color_mode(self, new_mode: str) -> None:
        if new_mode in RpiNeoPixelSPI.COLOR_MODES:
            self.__color_mode = new_mode
        else:
            raise ValueError(f"Error: Unknown color mode: '{new_mode}'")


    @property
    def brightness(self) -> float:
        return self.__brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        self.__brightness = float(np.clip(value, 0.0, 1.0))
        if self.__auto_write:
            self.show()


    @property
    def auto_write(self) -> bool:
        return self.__auto_write

    @auto_write.setter
    def auto_write(self, value:bool) -> None:
        if value and not self.__auto_write:
            self.show()
        self.__auto_write = value


    @property
    def num_pixels(self) -> int:
        """Get the number of pixels in the strip."""
        return self.__num_pixels


    def cleanup(self) -> None:
        """
        Clean up resources by closing the SPI device.
        Should be called when done using the NeoPixel strip.
        """
        if hasattr(self, '_spi') and self._spi is not None:
            try:
                self._spi.close()
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
            finally:
                self._spi = None

    def __enter__(self) -> 'RpiNeoPixelSPI':
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        self.cleanup()


def Demo():
    from time import sleep
    from random import randrange, random


    with RpiNeoPixelSPI(144, device=0, brightness=0.5, color_mode="HSV", pixel_order="GRB", gamma_func=gamma4g) as neo:
        with RpiNeoPixelSPI(8, device=1, color_mode="RGB", brightness=0.5) as neo1:
            neo1.set_value(0, 0.75, 1, 1, color_mode="HSV")
            neo1.set_value(1, [1, 0, 0])
            neo1.set_value(2, np.array([0, 1, 0]))
            neo1[3] = [0, 0, 1]
            neo1[4] = (1, 1, 0,)
            neo1[5] = np.array([1, 0, 1])
            neo1[6] = 0, 1, 1
            neo1[7] = 1, 1, 1
            neo1() # = neo.show()
            for i in range(neo.num_pixels):
                v = i/(neo.num_pixels-1)
                color = 1, 0, v
                neo(i, color) # immediate show() on this
            neo.set_value([23, 96, 120], [0, 0, 1], color_mode="RGB")
            sleep(1)


if __name__ == "__main__":
    Demo()
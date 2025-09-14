"""
Raspberry Pi NeoPixel driver using SPI interface.
Author: AlexL
License: MIT
Github:
"""

import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_yiq, yiq_to_rgb, rgb_to_hls, hls_to_rgb
from typing import Union, Callable
from numpy.typing import NDArray
from spidev import SpiDev


def gamma_square(value: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Applies a square gamma correction to the input value.
    Parameters:
        value (np.array or float): The input array of color values or a single float value.
    Returns:
        np.array or float: The gamma-corrected array or float value.
    """
    return np.clip(value**2, 0.0, 1.0)


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
    return a * x**4 + b * x**3 + c * x**2 + d * x


def lerp4(value: float, p0: float, p1: float, p2: float, p3: float, p4: float) -> float:
    """
    Linearly interpolates between five points (p0, p1, p2, p3, p4) based on the input value.
    The input value should be in the range [0, 1]. The function divides this range into four segments:
    [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1], and performs linear interpolation within the appropriate segment.
    """
    value = max(0, min(1, value))    
    if value < 0.25:
        t = value / 0.25  # Scale value to [0, 1] for the first segment
        return p0 * (1 - t) + p1 * t
    elif value < 0.5:
        t = (value - 0.25) / 0.25  # Scale value to [0, 1] for the second segment
        return p1 * (1 - t) + p2 * t
    elif value < 0.75:
        t = (value - 0.5) / 0.25  # Scale value to [0, 1] for the third segment
        return p2 * (1 - t) + p3 * t
    else:
        t = (value - 0.75) / 0.25  # Scale value to [0, 1] for the fourth segment
        return p3 * (1 - t) + p4 * t


def hue_lerp(value):
    p0 = 0.
    p1 = 0.2
    p2 = 0.5
    p3 = 0.8
    p4 = 1.0
    return lerp4(value, p0, p1, p2, p3, p4)


class RpiNeoPixelSPI:
    """    
    Driver for NeoPixel LEDs using SPI interface on a Raspberry PI.
    Args:
        num_pixels (int): Number of NeoPixel LEDs in the strip.
        device (int, optional): SPI device number. Defaults to 0.
            device=0 -> /dev/spidev0.0 uses BCM pins 8 (CE0), 9 (MISO), 10 (MOSI), 11 (SCLK)
            device=1 -> /dev/spidev0.1 uses BCM pins 7 (CE1), 9 (MISO), 10 (MOSI), 11 (SCLK)
        gamma_func (Callable, optional): Function to apply gamma correction. Defaults to gamma4g.
        hue_lerp (Callable, optional): Function to apply hue interpolation. Defaults to None.
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
    CLOCK_400KHZ        = 1_625_000
    CLOCK_800KHZ        = 3_250_000
    CLOCK_1200KHZ       = 6_500_000
    # Add constants for SPI encoding
    SPI_HIGH_BIT = 0xC0
    SPI_LOW_BIT = 0x80
    SPI_HIGH_BIT2 = 0x0C
    SPI_LOW_BIT2 = 0x08

    def __init__(self,
                num_pixels: int,
                *,
                device: int = 0,
                gamma_func: Callable | None = gamma4g,
                hue_lerp_func: Callable | None = None,
                color_mode: str = "HSV", 
                brightness: float = 1.0, 
                auto_write: bool= False,
                pixel_order: str = "GRB",
                clock_rate: int = CLOCK_800KHZ
                ) -> None:

        self.__pixel_order = pixel_order.upper()
        if 'R' in self.__pixel_order and 'G' in self.__pixel_order and 'B' in self.__pixel_order:
            pass
        else:
            raise ValueError(f"Unexpected pixel order: {self.__pixel_order}")

        self.__color_mode = color_mode.upper()
        if not self.__color_mode in RpiNeoPixelSPI.COLOR_MODES:
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
        self.__hue_lerp_func = hue_lerp_func
        self.__auto_write = auto_write
        self.__pixel_buffer = np.zeros((self.__num_pixels, len(self.__pixel_order)), dtype=float)

        # Pre-allocate SPI buffer
        bits_per_pixel = 16 if len(self.__pixel_order) == 4 else 12
        self.__spi_buffer = bytearray(bits_per_pixel * self.__num_pixels)


    def __to_RGB(self, value: np.ndarray, color_mode=None) -> np.ndarray:
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


    def __to_HSV(self, value: np.ndarray, color_mode=None) -> np.ndarray:
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
 

    def __from_RGB(self, rgb: np.ndarray, color_mode=None) -> np.ndarray:
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


    def __apply_gamma(self, rgbw:np.ndarray) -> list[int]:
        rgbw = self.__gamma_func(rgbw * self.__brightness)
        if rgbw.shape[0] == 3:
            rgbw = np.append(rgbw, 0.0)

        return np.clip(np.round(255 * rgbw), 0, 255).astype(int).tolist()


    def __encode_color_bits(self, color: int, msb_mask: int, c_mask: int) -> tuple[int, int]:
        """Encode two color bits into one byte SPI format."""
        bit1 = bool(color & msb_mask)
        color = (color << 1) & c_mask
        bit2 = bool(color & msb_mask)
        color = (color << 1) & c_mask
        byte = (self.SPI_HIGH_BIT if bit1 else self.SPI_LOW_BIT) | \
               (self.SPI_HIGH_BIT2 if bit2 else self.SPI_LOW_BIT2)
        return byte, color


    def _write_buffer(self, rgb_buffer: np.ndarray | None = None) -> None:
        """
        Write pixel data to NeoPixels using SPI protocol.
        
        Args:
            buffer: Optional numpy array containing pixel data. 
                   If None, uses internal pixel buffer.
        """
        if rgb_buffer is None:
            rgb_buffer = self.__pixel_buffer 

         # Set up bit masks based on RGB/RGBW mode
        if self._has_W:
            bits_per_pixel = 16  # 4 bytes * 4 bits
            msb_mask = 0x80000000
            c_mask = 0xFFFFFFFF
        else:
            bits_per_pixel = 12  # 3 bytes * 4 bits
            msb_mask = 0x800000
            c_mask = 0xFFFFFF

        byte_index = 0

        # Process each pixel
        for pixel_index in range(rgb_buffer.shape[0]):
            # Apply gamma correction and get color values

            r, g, b, w = self.__apply_gamma(rgb_buffer[pixel_index])

            # Combine color channels according to pixel order
            color = 0
            for channel in self.__pixel_order:
                color = color << 8
                match channel:
                    case 'R': color |= r
                    case 'G': color |= g
                    case 'B': color |= b
                    case 'W': color |= w

            # Convert to SPI format
            for bit_index in range(bits_per_pixel):
                byte, color = self.__encode_color_bits(color, msb_mask, c_mask)
                self.__spi_buffer[byte_index] = byte
                byte_index += 1

        # Send data to device
        self._spi.writebytes2(self.__spi_buffer) # type: ignore


    def __setitem__(self, index: int, value: np.ndarray) -> None:
        rgb = self.__to_RGB(np.clip(value, 0.0, 1.0))

        if rgb.shape[0] == 3 and self._has_W:
            rgb = np.append(rgb, self.__pixel_buffer[index][3])

        self.__pixel_buffer[index] = rgb
        if self.__auto_write:
            self.show()


    def __getitem__(self, index: int) -> np.ndarray:
        return self.__from_RGB(self.__pixel_buffer[index])


    def fill(self, value:np.ndarray, color_mode=None) -> None:
        rgb = self.__to_RGB(value)
        self.__pixel_buffer [:] = rgb
        if self.__auto_write:
            self.show()


    def fill_hsv(self, *, hue:float|None=None, sat:float|None=None, val:float|None=None, white:float|None=None) -> None:

        raise NotImplementedError("fill_hsv is currently not working correctly. If you like to contribute...")

        if hue is None and sat is None and val is None and white is None:
            return
        
        for i in range(self.num_pixels):
            hsv = self.__to_HSV(self.__pixel_buffer[i])
            if hue is not None:
                hsv[0] = hue
            if sat is not None:
                hsv[1] = sat
            if val is not None:
                hsv[2] = val
            rgb = self.__to_RGB(hsv, color_mode="HSV")
            if white is not None and self._has_W:
                rgb[3] = white
            self.__pixel_buffer[i] = rgb

        if self.__auto_write:
            self.show()


    def clear(self) -> None:
        """Clear all pixels by setting them to black."""
        black = self.COLOR_RGB_BLACK_W if self._has_W else self.COLOR_RGB_BLACK
        self.fill(black)


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
            case 2: # set pixel at index
                self[args[0]] = args[1]
            case _:
                raise ValueError("Too many arguments!")
            
        if not self.__auto_write:
            self.show()


    def set_value(self, index:int, a:float, b:float, c:float, w:float=0.) -> None:
        if self._has_W:
            self[index] = np.array([a, b, c, w])
        else:
            self[index] = np.array([a, b, c])


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
                self.clear()  # Turn off all pixels
                self.show()
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


def Test():
    from time import sleep
    from random import randrange

    with RpiNeoPixelSPI(144, device=0, brightness=0.25, color_mode="HSV", pixel_order="GRB", gamma_func=gamma4g) as neo:

        #neo[0] = np.array([1., 0., 0.])
        #neo[1] = np.array([0., 1., 0.])
        #neo.set_value(2, 0., 0., 1.)
        #neo[3] = np.array([1., 1, 0.])
        #neo()

        while True:
            for i in range(neo.num_pixels):
                v = i/(neo.num_pixels-1)
                color = np.array([0.9,1,v])
                neo[i] = color
            neo.show()
                # sleep(0.1)
            while True:
                sleep(0.5)
            neo.clear()
            neo.show()
            sleep(0.5)


if __name__ == "__main__":
    Test()
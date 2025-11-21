"""
Raspberry Pi NeoPixel driver using SPI interface.
Author: AlexL
License: MIT
Github: https://github.com/Hangover3832/rpi_neopixel_spi
"""

import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_yiq, yiq_to_rgb, rgb_to_hls, hls_to_rgb
from typing import Callable
from devices import SpiDev, OutputDevice # type: ignore
from devices import Spi_Clock
from colors import PixelOrder, ColorMode, default_gamma


class RpiNeoPixelSPI:
    """    
    Driver for NeoPixel LEDs using SPI interface on a Raspberry PI.
    Args:
        num_pixels (int): Number of NeoPixel LEDs in the strip.
        device (int, optional): SPI device number. Defaults to 0.
            device=0 -> /dev/spidev0.0 uses BCM pins 8 (CE0), 9 (MISO), 10 (MOSI), 11 (SCLK)
            device=1 -> /dev/spidev0.1 uses BCM pins 7 (CE1), 9 (MISO), 10 (MOSI), 11 (SCLK)
        gamma_func (Callable, optional): Function to apply gamma correction. Defaults to default_gamma.
        color_mode (ColorMode, optional): Color mode for input values. Options are RGB, HSV, YIQ, HLS. Defaults to HSV.
        brightness (float, optional): Brightness level (0.0 to 1.0). Defaults to 1.0.
        auto_write (bool, optional): If True, updates the LEDs automatically on value change. Defaults to False.
        pixel_order (PixelOrder, optional): Order of color channels in the NeoPixel (e.g., GRB, RGBW). Defaults to GRB.
        clock_rate (Spi_Clock, optional): SPI clock rate in Hz. Defaults to CLOCK_800KHZ.

    Clock Rates:
        - Spi_Clock.CLOCK_400KHZ: Standard rate for WS2812 pixels
        - Spi_Clock.CLOCK_800KHZ: High-speed rate for WS2812B pixels
        - Spi_Clock.CLOCK_1200KHZ: Maximum rate for some pixels
    """

    COLOR_RGB_BLACK     = np.array([0., 0., 0.])
    COLOR_RGB_BLACK_W   = np.array([0., 0., 0., 0.])
    COLOR_RGB_WHITE     = np.array([1., 1., 1.])
    COLOR_RGB_WHITE_W   = np.array([0., 0., 0., 1.])
    SPI_HIGH_BIT        = 0xC0
    SPI_LOW_BIT         = 0x80
    SPI_HIGH_BIT2       = 0x0C
    SPI_LOW_BIT2        = 0x08


    def __init__(self,
                num_pixels: int,
                *,
                device: int = 0,
                gamma_func: Callable | None = None,
                color_mode: ColorMode = ColorMode.HSV,
                brightness: float = 1.0, 
                auto_write: bool = False,
                pixel_order: PixelOrder = PixelOrder.GRB,
                clock_rate: Spi_Clock = Spi_Clock.CLOCK_800KHZ,
                custom_cs: int | None = None,
                ) -> None:

        self.__pixel_order: PixelOrder = pixel_order
        self.__color_mode: ColorMode = color_mode
        self.__auto_write = auto_write

        if custom_cs is not None: # use a custom chip select (cs) BCM pin
            self._cs = OutputDevice(custom_cs, active_high=False, initial_value=True)
            device = 0
        else:
            self._cs = None

        if device not in [0, 1]:
            raise ValueError("Error: device must be 0 or 1")

        try:
            self._spi = SpiDev(pixel_order) if hasattr(SpiDev, "IS_DUMMY_DEVICE") else SpiDev()
            self._spi.open(bus=0, device=device)
            self._spi.max_speed_hz = clock_rate.value
            self._spi.mode = 0
            self._spi.bits_per_word = 8
            if self._cs is not None:
                self._spi.no_cs = True
        except OSError: # catching a possible SpiDev.no_cs error as the rasbian kernel driver might not suppoprt it
            pass # in this case, the default cip select signal on pin 8 is still handled by the driver beside the user defined pin
        except: 
            raise RuntimeError("Error: Could not open SPI device. Ensure SPI is enabled in raspi-config and the device number is correct.")

        self.__brightness: float = float(np.clip(brightness, 0., 1.))
        self.__gamma_func: Callable = gamma_func or default_gamma

        if (num_led := len(self.__pixel_order.name)) == 4:
            self._double_bits_per_pixel = 16
            self.__msb_mask = 0x80000000
            self.__c_mask = 0xFFFFFFFF
        else:
            self._double_bits_per_pixel = 12
            self.__msb_mask = 0x800000
            self.__c_mask = 0xFFFFFF

        self.__pixel_buffer = np.zeros((num_pixels, num_led), dtype=np.float32)

        # Pre-allocate buffer for the encoded bits
        self.__spi_buffer = np.zeros([self._double_bits_per_pixel, self.num_pixels], dtype=np.uint8)


    def _from_RGB(self, rgb: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Convert value from RGB to color_mode (or, if not provided, to the current color mode)"""

        result = {
            ColorMode.RGB: rgb[0:3],
            ColorMode.HSV: np.array(rgb_to_hsv(*rgb[0:3])),
            ColorMode.YIQ: np.array(rgb_to_yiq(*rgb[0:3])),
            ColorMode.HLS: np.array(rgb_to_hls(*rgb[0:3]))
        }[color_mode or self.__color_mode]

        return np.append(result, rgb[3]) if rgb.shape[0] > 3 and self._has_W else result


    def _to_RGB(self, value: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Convert value from color_mode (or, if not provided, the current color mode) to RGB"""

        result = {
            ColorMode.RGB: value[0:3],
            ColorMode.HSV: np.array(hsv_to_rgb(*value[0:3])),
            ColorMode.YIQ: np.array(yiq_to_rgb(*value[0:3])),
            ColorMode.HLS: np.array(hls_to_rgb(*value[0:3]))
        }[color_mode or self.__color_mode]

        return np.append(result, value[3]) if value.shape[0] > 3 and self._has_W else result


    def _to_HSV(self, value: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Convert value from color_mode (or, if not provided, the current color mode) to HSV"""

        color_mode = color_mode or self.__color_mode
        if color_mode == ColorMode.HSV:
            return value

        rgb = self._to_RGB(value, color_mode)
        hsv = np.array(rgb_to_hsv(*rgb[0:3]))

        return np.append(hsv, value[3]) if value.shape[0] > 3 and self._has_W else hsv


    def _write_buffer(self, a_rgb_buffer: np.ndarray | None = None) -> None:
        """
        Write pixel data to NeoPixels using SPI protocol.
        
        Args:
            buffer: Optional numpy array containing pixel data. 
                   If None, uses internal pixel buffer.
        """

        rgb_buffer = a_rgb_buffer.copy() if a_rgb_buffer else self.__pixel_buffer.copy()

        # Apply brightness and gamma correction, scale to [0, 255], and convert to uint8
        rgb_buffer = 255 * self.__gamma_func(rgb_buffer * self.__brightness)
        rgb_buffer = np.clip(np.round(rgb_buffer), 0, 255).astype(np.uint8)

        # rows are now pixels, columns are R,G,B,(W)
        # rearange the rgb_buffer to the correct pixel order
        # Here, we allow every possible pixel order with R,G,B and optional W
        rgb_buffer = rgb_buffer[:, [self.__pixel_order.name.index(c) for c in 'RGBW' if c in self.__pixel_order.name]]

        # Convert [r, g, b, (w)] uint8 to a single uint32:
        if self._has_W:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00_00, 0x1_00_00, 0x1_00, 1], dtype=np.uint32)
        else:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00, 0x1_00, 1], dtype=np.uint32)

        # reduce the array to a single uint32 per pixel:
        rgb_buffer = np.bitwise_or.reduce(rgb_buffer, axis=1, dtype=np.uint32)

        # shift out 2 bits of each pixel and encode them to a byte for SPI transmission:
        for i in range(self._double_bits_per_pixel):
            bit1 = (rgb_buffer & self.__msb_mask).astype(bool)
            rgb_buffer = ((rgb_buffer << 1) & self.__c_mask).astype(np.uint32)
            bit2 = (rgb_buffer & self.__msb_mask).astype(bool) 
            rgb_buffer = ((rgb_buffer << 1) & self.__c_mask).astype(np.uint32)
            # encode 2 pixel bits into 1 SPI byte:
            self.__spi_buffer[i] = (np.where(bit1, self.SPI_HIGH_BIT, self.SPI_LOW_BIT) | np.where(bit2, self.SPI_HIGH_BIT2, self.SPI_LOW_BIT2)).astype(np.uint8)

        # Send data to device:
        if self._cs is not None:
            self._cs.on() # chip enable
        self._spi.writebytes2(self.__spi_buffer.T.flatten()) # type: ignore
        if self._cs is not None:
            self._cs.off() # chip disable


    def __setitem__(self, index: int | slice, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """Indexed or sliced Beopixel access"""
        self.set_value(index, value)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._from_RGB(self.__pixel_buffer[index])
    
    def __len__(self) -> int:
        """Get the number of pixels in the strip."""
        return self.__pixel_buffer.shape[0]


    def _write_value_to_buffer(self, indexes: int | list[int] | tuple[int] | slice, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        value = np.array(value)
        if value.shape[0] == 3 and self._has_W:
            # if RGB is passed but the stripe has RGBW, only store RGB and keep W as is
            self.__pixel_buffer[indexes, :3] = value
        else:
            self.__pixel_buffer[indexes] = value


    def set_value(self, indexes: int | list[int] | tuple[int] | slice, value: np.ndarray | list[float] | tuple[float, ...], color_mode: ColorMode | None = None) -> 'RpiNeoPixelSPI':
        rgb = self._to_RGB(np.clip(value, 0., 1.), color_mode=color_mode)
        self._write_value_to_buffer(indexes, rgb)
        return self.show() if self.__auto_write else self


    def fill(self, value: np.ndarray | list | tuple, color_mode: ColorMode | None = None) -> 'RpiNeoPixelSPI':
        return self.set_value(slice(None), value=value, color_mode=color_mode)


    def __iadd__(self, value: np.ndarray | float) ->'RpiNeoPixelSPI':
        """Add value to the pixel buffer in RGB space, e.g. pixels += 0.1"""
        self.__pixel_buffer =  np.clip(self.__pixel_buffer + value, 0., 1.)
        return self.show() if self.__auto_write else self

    def __imul__(self, value: np.ndarray | float) ->'RpiNeoPixelSPI':
        """Multiply value with the pixel buffer in RGB space, e.g. neo *= 0.9"""
        self.__pixel_buffer =  np.clip(self.__pixel_buffer * value, 0., 1.)
        return self.show() if self.__auto_write else self

    def __lshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the left by amount, e.g. `pixels << 1`"""
        return self.roll(int(-abs(amount)))

    def __rshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the right by amount, e.g. `pixels >> 1`"""
        return self.roll(int(abs(amount)))

    def __ilshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the left by amount, e.g. `pixels <<= 1`"""
        self.roll(int(-abs(amount)))
        return self if self.auto_write else self.show()
    
    def __irshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the right by amount, e.g. `pixels >>= 1`"""
        self.roll(int(abs(amount)))
        return self if self.auto_write else self.show()
    
    def __invert__(self)-> 'RpiNeoPixelSPI':
        """Invert all colors of all pixels, e.g. `~pixels` """
        self.__pixel_buffer = 1.0 - self.__pixel_buffer
        return self.show() if self.auto_write else self


    def clear(self) -> 'RpiNeoPixelSPI':
        """Clear all pixels by setting them to black."""
        return self.fill(self.blank, color_mode=ColorMode.RGB)


    def show(self) -> 'RpiNeoPixelSPI':
        """Update the NeoPixels with the current pixel buffer."""
        self._write_buffer()
        return self


    def roll(self, shift: int = 1, value: np.ndarray | list[float] | tuple[float, ...] | None = None) -> 'RpiNeoPixelSPI':
        """Roll the pixel buffer by the specified shift amount.

        Args:
            shift (int): Number of positions to shift. Positive values shift right, negative values shift left.
            value: If value is None, pixels that roll off one end will reappear at the other end. If a value is provided, 
            pixels that roll in will ne set to this value.
        """

        if value is None:
            self.__pixel_buffer = np.roll(self.__pixel_buffer, shift, axis=0)
        else:
            value = np.array(value)

            if shift > 0:
                self.__pixel_buffer[shift:] = self.__pixel_buffer[:-shift]
                self._write_value_to_buffer(slice(None,shift), value)

            elif shift < 0:
                self.__pixel_buffer[:shift] = self.__pixel_buffer[-shift:]
                self._write_value_to_buffer(slice(shift, None), value)

        return self.show() if self.__auto_write else self


    def __call__(self,
                 indexes: int | list[int] | tuple[int] | slice | None = None, 
                 value: np.ndarray | list[float] | tuple[float, ...] | None = None
                 ) -> 'RpiNeoPixelSPI':
        # immediate update
        if indexes is None:
            return self.show()

        if value is None:
            return self.set_value(indexes, self.blank)

        return self.set_value(indexes, value)


    @property
    def blank(self) -> np.ndarray:
        """Get a black color value appropriate for the pixel type (RGB or RGBW)."""
        return self.COLOR_RGB_BLACK_W if self._has_W else self.COLOR_RGB_BLACK

    @property
    def _has_W(self) -> bool:
        return self.__pixel_buffer.shape[1] > 3

    @property
    def gamma_func(self) -> Callable[[float], float]:
        return self.__gamma_func

    @gamma_func.setter
    def gamma_func(self, new_gamma: Callable[[float], float]) -> None:
        self.__gamma_func = new_gamma
        if self.__auto_write:
            self.show()

    @property 
    def color_mode(self) -> ColorMode:
        return self.__color_mode

    @color_mode.setter
    def color_mode(self, new_mode: ColorMode) -> None:
        self.__color_mode = new_mode

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
    def CS(self) -> OutputDevice | None:
        return self._cs


    def cleanup(self) -> None:
        """
        Clean up resources by closing the devices.
        Should be called when done using the NeoPixel strip.
        """
        # self.clear()()
        if hasattr(self._spi, "IS_DUMMY_DEVICE"):
            print()

        if hasattr(self, '_cs') and self._cs is not None:
            try:
                self._cs.close()
            except Exception as e:
                print(f"Warning: Error during cleanup (pin): {e}")
            finally:
                self._cs = None

        if hasattr(self, '_spi') and self._spi is not None:
            try:
                self._spi.close()
            except Exception as e:
                print(f"Warning: Error during cleanup (spi): {e}")
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


def class_test():
    from time import sleep

    with RpiNeoPixelSPI(100, color_mode=ColorMode.RGB, pixel_order=PixelOrder.GRBW, brightness=1) as neo:
        neo.fill((1,1,1,0)) # fill white on black
        neo([1,10,20,99], [0,0,0,1]) # set pixels 1, 10, 20 and 99 to black on white
        neo[30:40] = 0, 0, 1, 1 # set pixels 30..39 to blue on white
        neo[50] = 1, 0, 0 # set pixel 50 to red
        neo[60] = 0, 1, 0 # set pixel 60 to green
        neo() # show()
        for _ in range(10):
            neo << 1 # left roll by 1, but not show() # type: ignore
            sleep(0.5)
            (~neo)() # invert colors and show()
            neo *= 0.9 # multiply all led values with 0.9
        for _ in range(10):
            neo += np.array([0,0,0,0.1]) # increase white LED by 0.1
            neo >>= 2 # right roll by 2 and show()
            sleep(0.5)


def GammaTest() -> None:
    with RpiNeoPixelSPI(144, device=0, brightness=0.25, color_mode=ColorMode.HSV) as neo:
        for i in range(neo.num_pixels):
            v = i/(neo.num_pixels-1)
            color = 1, 0, v
            neo[i] = color
        neo()


def Rainbow():
    from time import sleep
    from colors import linear_gamma
    with RpiNeoPixelSPI(144, pixel_order=PixelOrder.GRB, gamma_func=linear_gamma, brightness=1) as neo:
        neo.clear()() # clear() and show()
        for i in range(neo.num_pixels):
            v = i/(neo.num_pixels-1)
            color = v, 1., 1.
            neo[i] = color
        while True:
            neo >>= 1 # roll() and show()
            sleep(0.02)


if __name__ == "__main__":
    class_test()
    Rainbow()
    # GammaTest()

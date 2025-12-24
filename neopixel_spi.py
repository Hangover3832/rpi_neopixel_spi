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
from colors import PixelOrder, ColorMode, G


PixelIndex = int | list[int] | tuple[int, ...] | slice
PixelValue = np.ndarray | list[float] | tuple[float, ...] | float | int

class RpiNeoPixelSPI:
    """    
    Driver for NeoPixel LEDs using SPI interface on a Raspberry PI.
    Args:
        num_pixels (int): Number of NeoPixel LEDs in the strip.
        device (int, optional): SPI device number. Defaults to 0.
            - device=0 -> /dev/spidev0.0 uses BCM pins 8 (CE0), 9 (MISO), 10 (MOSI), 11 (SCLK)
            - device=1 -> /dev/spidev0.1 uses BCM pins 7 (CE1), 9 (MISO), 10 (MOSI), 11 (SCLK)
        gamma_func (Callable, optional): Function to apply gamma correction. Defaults to default_gamma.
        color_mode (ColorMode, optional): Color mode for input values. Options are RGB, HSV, YIQ, HLS. Defaults to HSV.
        brightness (float, optional): Brightness level (0.0 to 1.0). Defaults to 1.0.
        auto_write (bool, optional): If True, updates the LEDs automatically on value change. Defaults to False.
        pixel_order (PixelOrder, optional): Order of color channels in the NeoPixel (e.g., GRB, RGBW). Defaults to GRB.
        clock_rate (Spi_Clock, optional): SPI clock rate in Hz. Defaults to CLOCK_800KHZ.
        custom_cs (int, optional): Custom chip select pin.
            **Note: if you use custom chip select, the Rabian driver might not be able to to disable the standart cs pin.
            Please leave the standard pin (7 or 8) unconnected in this case, especial if you want to use multple strips at the same time!**
        max_power (float, optional): Maximum power in watts. The strip is dimmed automatically if this limit is reached.
            Set this value to 0 to disable power control.
            The calculation of the power consumtion can be adjusted with the `wats_per_led` attribute.

    Clock Rates:
        - Spi_Clock.CLOCK_400KHZ: Standard rate for WS2812 pixels
        - Spi_Clock.CLOCK_800KHZ: High-speed rate for WS2812B pixels
        - Spi_Clock.CLOCK_1200KHZ: Maximum rate for some pixels

    Example:
        ```
        # Set all pixel to red (HSV mode by default)
        neo = RpiNeoPixelSPI(50)
        neo[:] = (0.0, 1.0, 1.0))
        neo()
        ```
    """

    SPI_HIGH_BIT        = 0xC0
    SPI_LOW_BIT         = 0x80
    SPI_HIGH_BIT2       = 0x0C
    SPI_LOW_BIT2        = 0x08


    def __init__(self,
                num_pixels: int,
                *,
                device: int = 0,
                gamma_func: Callable = G.default.value,
                color_mode: ColorMode = ColorMode.HSV,
                brightness: float = 1.0, 
                auto_write: bool = False,
                pixel_order: PixelOrder = PixelOrder.GRB,
                clock_rate: Spi_Clock = Spi_Clock.CLOCK_800KHZ,
                custom_cs: int | None = None,
                max_power: float = 0.0
                ) -> None:

        self.reversed: bool = False
        self.watts_per_led: np.ndarray = np.array([0.081, 0.081, 0.08, 0.09])
        self._pixel_order: PixelOrder = pixel_order
        self._color_mode: ColorMode = color_mode
        self._auto_write = auto_write
        self._num_lit_pixels: int = 0
        self._current_power: float = 0.0
        self._max_power: float = max_power
        self._index: int = 0

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

        self._brightness: float = float(np.clip(brightness, 0., 1.))
        self._gamma_func: Callable = gamma_func

        if self._pixel_order.num == 4:
            self._double_bits_per_pixel = 16
            self._msb_mask = 0x80000000
            self._c_mask = 0xFFFFFFFF
        else:
            self._double_bits_per_pixel = 12
            self._msb_mask = 0x800000
            self._c_mask = 0xFFFFFF

        self._pixel_buffer = np.zeros((num_pixels, self._pixel_order.num), dtype=np.float32)

        # Pre-allocate buffer for the encoded bits
        self._spi_buffer = np.zeros([self._double_bits_per_pixel, self.num_pixels], dtype=np.uint8)

        self._mini_screens: list[np.ndarray] = []


    def add_virtual_screen(self, config: np.ndarray) -> int:
        """
        Add a virtual 2 dimensional screen area to the Neopixel stripe.
        
        :param config: A 2-dimensional array that contains all the pixel indices that bild up the screen,
        mapping from the top left to the bottom right on the virtual screen.
        :type config: np.ndarray[[int, ...], ...]
        :returns: The index number of the newly created screen.
        :rtype: int
        """

        self._mini_screens.append(config.astype(np.int16))
        return len(self._mini_screens)-1


    def virtual_screen_data(self, index: int, data: np.ndarray, color_mode: ColorMode  | None = None) -> 'RpiNeoPixelSPI':
        """
        Put pixel data onto a virtual screen.

        :param index: The index of the virtuel screen created and returned by `add_virtual_screen()`.
        :type index: int
        :param data: A 3-dimensional array that contains RGB(W) Pixel values to put on the virtual screen.
        Note that the shape of the `data` array must match the shape of the virtual screen.
        :type data: np.ndarray[[[float, ...]]]
        :returns: self
        :rtype: RpiNeoPixelSPI
        """

        assert data.shape[:2] == self._mini_screens[index].shape
        indices = self._mini_screens[index].flatten()
        data = data.reshape(-1, data.shape[2]).squeeze()
        for d, i in enumerate(indices):
            self.set_value(i, data[d], color_mode=color_mode)

        return self.show() if self._auto_write else self


    def _from_RGB(self, rgb: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Convert value from RGB to color_mode (or, if not provided, to the current color mode)"""
        result = (color_mode or self._color_mode).from_rgb(rgb)
        return np.append(result, rgb[3]) if rgb.shape[0] > 3 and self._has_W else result

    def _to_RGB(self, value: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Convert value from color_mode (or, if not provided, the current color mode) to RGB"""
        result = (color_mode or self._color_mode).to_rgb(value)
        return np.append(result, value[3]) if value.shape[0] > 3 and self._has_W else result

    def _to_HSV(self, value: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Convert value from color_mode (or, if not provided, the current color mode) to HSV"""
        result = (color_mode or self._color_mode).to_hsv(value)
        return np.append(result, value[3]) if value.shape[0] > 3 and self._has_W else result


    def _write_buffer(self) -> None:
        """Write pixel data to NeoPixels using SPI protocol."""

        rgb_buffer = self._pixel_buffer.copy()[::-1] if self.reversed else self._pixel_buffer.copy()

        # Apply brightness and gamma correction
        rgb_buffer = np.clip(self._gamma_func(rgb_buffer * self._brightness), 0.0, 1.0)

        # calculate power consumption
        watts = self.watts_per_led if self._has_W else self.watts_per_led[:3]
        self._current_power = np.sum(watts * rgb_buffer)  

        # Power consumption limiter
        if (self._max_power > 1e-6) and (self._current_power > self._max_power):
            rgb_buffer *= self._max_power/self._current_power
            self._current_power = self._max_power

        # scale to [0, 255], and convert to uint8:
        rgb_buffer = np.clip(np.round(255 * rgb_buffer), 0, 255).astype(np.uint8)
        self._num_lit_pixels = int(np.count_nonzero(np.max(rgb_buffer, axis=1)))

        # rows are now pixels, columns are R,G,B,(W)
        # rearange the rgb_buffer to the correct PixelOrder
        # Here, we allow every possible pixel order with R,G,B and optional W
        rgb_buffer = rgb_buffer[:, [self._pixel_order.name.index(c) for c in 'RGBW' if c in self._pixel_order.name]]

        # Convert [r, g, b, (w)] to uint32:
        if self._has_W:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00_00, 0x1_00_00, 0x1_00, 1], dtype=np.uint32)
        else:
            rgb_buffer = rgb_buffer * np.array([0x1_00_00, 0x1_00, 1], dtype=np.uint32)

        # reduce the array to a single uint32 per pixel:
        rgb_buffer = np.bitwise_or.reduce(rgb_buffer, axis=1, dtype=np.uint32)

        # shift out 2 bits of each pixel and encode them to a byte for SPI transmission:
        for i in range(self._double_bits_per_pixel):
            bit1 = (rgb_buffer & self._msb_mask).astype(bool)
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            bit2 = (rgb_buffer & self._msb_mask).astype(bool) 
            rgb_buffer = ((rgb_buffer << 1) & self._c_mask).astype(np.uint32)
            # encode 2 pixel bits into 1 SPI byte:
            self._spi_buffer[i] = (np.where(bit1, self.SPI_HIGH_BIT, self.SPI_LOW_BIT) | np.where(bit2, self.SPI_HIGH_BIT2, self.SPI_LOW_BIT2)).astype(np.uint8)

        # Send data to device:
        if self._cs is not None:
            self._cs.on() # chip enable
        self._spi.writebytes2(self._spi_buffer.T.flatten()) # type: ignore
        if self._cs is not None:
            self._cs.off() # chip disable


    def __setitem__(self, index: int | slice, value: PixelValue) -> None:
        """Indexed or sliced Neopixel access"""
        self.set_value(index, value)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._from_RGB(self._pixel_buffer[index])

    def __len__(self) -> int:
        """Get the number of pixels in the strip."""
        return self._pixel_buffer.shape[0]


    def _write_value_to_buffer(self, index: PixelIndex, value: PixelValue) -> None:
        """Write RGB value(s) to the interbal pixelbuffer that is held in RGB space"""

        if isinstance(value, (float, int)):
            # a simple number applies to the white pixel only if available
            if self._has_W:
                self._pixel_buffer[index, 3] = value
                return
            else:
                raise ValueError("Cannot set white LED on non RGBW Neopixel")

        if (value := np.array(value)).shape[0] == 3:
            # if RGB is passed but the stripe has RGBW, only store RGB and keep W as is
            self._pixel_buffer[index, :3] = value
            return

        self._pixel_buffer[index] = value[:4]


    def set_temperature(self, index:PixelIndex, temperature:float, brightness:float = 1.0) -> 'RpiNeoPixelSPI':
        """Set pixel value at index using an approximation for the black body temperature radiation, 
        maintaining a constant brightness. The temperature ranges from [0.0 .. 1.0]"""

        self._write_value_to_buffer(index, 
                                    brightness * self._color_mode.kelvin_to_rgb(
                                    np.clip(temperature, 0.0, 1.0)))
        return self.show() if self._auto_write else self


    def set_value(self, index: PixelIndex, value: PixelValue, color_mode: ColorMode | None = None) -> 'RpiNeoPixelSPI':
        """Set pixel value at index, use color_mode if specified else use the current color mode"""

        if isinstance(value, (float, int)):
            self._write_value_to_buffer(index, value)
        else:
            rgb = self._to_RGB(np.array(value), color_mode=color_mode)
            self._write_value_to_buffer(index, rgb)

        return self.show() if self._auto_write else self
    

    def next_(self, value: PixelValue, color_mode: ColorMode | None = None) -> int:
        """Set the value for the next pixel in the iteration"""
        self.set_value(result := next(self), value=value, color_mode=color_mode)
        return result

    def fill(self, value: PixelValue, color_mode: ColorMode | None = None) -> 'RpiNeoPixelSPI':
        """Fill all pixels with a given value"""
        return self.set_value(slice(None), value=value, color_mode=color_mode)

    def __iadd__(self, value: np.ndarray | float) ->'RpiNeoPixelSPI':
        """Add value to the pixel buffer in RGB space, e.g. pixels += 0.1"""
        self._pixel_buffer =  np.clip(self._pixel_buffer + value, 0., 1.)
        return self.show() if self._auto_write else self

    def __imul__(self, value: np.ndarray | float) ->'RpiNeoPixelSPI':
        """Multiply value with the pixel buffer in RGB space, e.g. neo *= 0.9"""
        self._pixel_buffer = np.clip(self._pixel_buffer * value, 0., 1.)
        return self.show() if self._auto_write else self

    '''
    def __lshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the left by amount, e.g. `pixels << 1`"""
        return self.roll(-int(abs(amount)))

    def __rshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the right by amount, e.g. `pixels >> 1`"""
        return self.roll(int(abs(amount)))
    '''

    def __ilshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the left by amount, e.g. `pixels <<= 1`"""
        return self.roll(-int(abs(amount)))
    
    def __irshift__(self, amount: int) -> 'RpiNeoPixelSPI':
        """roll to the right by amount, e.g. `pixels >>= 1`"""
        return self.roll(int(abs(amount)))

    def __invert__(self)-> 'RpiNeoPixelSPI':
        """Invert all colors of all pixels, e.g. `~pixels` """
        self._pixel_buffer = 1.0 - self._pixel_buffer
        return self.show() if self.auto_write else self


    def clear(self) -> 'RpiNeoPixelSPI':
        """Clear all pixels by setting them to black."""
        return self.fill(self.blank, color_mode=ColorMode.RGB)


    def show(self) -> 'RpiNeoPixelSPI':
        """
        Update the NeoPixels with the current pixel buffer.
        
        :return: The current instance of RpiNeoPixelSPI.
        :rtype: RpiNeoPixelSPI
        
        """
        self._write_buffer()
        return self


    def roll(self, shift: int = 1, value: PixelValue | None = None) -> 'RpiNeoPixelSPI':
        """
        Roll the pixel buffer by the specified shift amount.
        
        :param shift: Number of positions to shift. Positive values shift right, negative values shift left. Defaults to 1.
        :type shift: int
        :param value: If `value` is None, pixels that roll off one end will reappear at the other end. If a `value` is provided, 
            pixels that roll in will be set to this value. If `value` is a single number, only the white LED is affected in a RGBW stripe.
        :type value: PixelValue | None
        :returns: The current instance of RpiNeoPixelSPI.
        :rtype: RpiNeoPixelSPI
        """
        if shift == 0:
            return self

        if value is None:
            self._pixel_buffer = np.roll(self._pixel_buffer, shift, axis=0)
            return self.show() if self._auto_write else self
        else:
            if not isinstance(value, (float, int)):
                value = np.array(value)

            if shift > 0:
                self._pixel_buffer[shift:] = self._pixel_buffer[:-shift]
                return self.set_value(slice(None,shift), value)
            else:
                self._pixel_buffer[:shift] = self._pixel_buffer[-shift:]
                return self.set_value(slice(shift, None), value)


    def __call__(self, index: PixelIndex | None = None, value: PixelValue | None = None) -> 'RpiNeoPixelSPI':
        """
        Calls `show()` that updates the stripe if no index or value is provided. If index is provided but no value,
        the pixel(s) at index gets cleared. If both index and value are provided, the pixel
        at index is set to the specified value. If index is a slice, the pixels at that
        range are set to the specified value.

        :param index: Pixel index number(s)
        :type index: int or slice
        :param value: Pixel value. 
            If value is a number, it affects the White LED only in a RGBW stripe. A non RGBW stripe
            will throw an exception in this case.
        :type value: number or array like
        """

        if index is None:
            return self.show()

        if value is None:
            return self.set_value(index, self.blank)

        return self.set_value(index, value)


    def __iter__(self):
        return self

    def __next__(self):
        """ Iterate over the indices of the pixels. """
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        self._index += 1
        return self._index - 1


    @property
    def blank(self) -> np.ndarray:
        """Get a black color value appropriate for the pixel type (RGB or RGBW)."""
        return self._pixel_order.blank
    
    @property
    def _has_W(self) -> bool:
        """Check if the pixel type has a white channel."""
        return self._pixel_buffer.shape[1] > 3

    @property
    def gamma_func(self) -> Callable[[float], float]:
        """Get the gamma function used for color correction."""
        return self._gamma_func

    @gamma_func.setter
    def gamma_func(self, new_gamma: Callable[[float], float]) -> None:
        """Set a new gamma function for color correction."""
        self._gamma_func = new_gamma
        if self._auto_write:
            self.show()

    @property 
    def color_mode(self) -> ColorMode:
        """Get the current color mode."""
        return self._color_mode

    @color_mode.setter
    def color_mode(self, new_mode: ColorMode) -> None:
        """Set a new color mode."""
        self._color_mode = new_mode

    @property
    def brightness(self) -> float:
        """Get the current brightness level."""
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        """Set a new brightness level."""
        self._brightness = float(np.clip(value, 0.0, 1.0))
        if self._auto_write:
            self.show()

    @property
    def auto_write(self) -> bool:
        """Get the current auto_write state."""
        return self._auto_write

    @auto_write.setter
    def auto_write(self, value:bool) -> None:
        """Set a new auto_write state."""
        self._auto_write = value

    @property
    def CS(self) -> OutputDevice | None:
        """Get the current chip select pin."""
        return self._cs
    
    @property
    def num_pixels(self) -> int:
        """Get the number of pixels in the strip."""
        return self.__len__()

    @property 
    def num_lit_pixels(self) -> int:
        """Get the number of lit pixels in the strip."""
        return self._num_lit_pixels

    @property
    def power_consumption(self) -> float:
        """Returns the total power consumption [0..1]"""
        return self._current_power
    
    @property
    def max_power(self) -> float | None:
        return self._max_power

    @max_power.setter
    def max_power(self, max_power: float) -> None:
        self._max_power = max_power
        self._write_buffer()

    @property
    def is_simulated(self) -> bool:
        return hasattr(self._spi, "IS_DUMMY_DEVICE")
    
    @property
    def pixel_buffer(self) -> np.ndarray:
        return self._pixel_buffer

    @pixel_buffer.setter
    def pixel_buffer(self, value:np.ndarray) -> None:
        self._pixel_buffer = value


    def cleanup(self) -> None:
        """
        Clean up resources by closing the devices.
        Should be called when done using the NeoPixel strip.
        """
        # self.clear()()
        if hasattr(self._spi, 'IS_DUMMY_DEVICE'):
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

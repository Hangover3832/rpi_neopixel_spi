**This is a Python and GIT learning project**

# NeoPixel SPI Driver Library for the Raspberry Pi

This Python library provides an optimized SPI driver for controlling WS2812/SK6812 (NeoPixel) LED strips. It features efficient bit encoding that allows controlling up to 320 LEDs in a single SPI buffer transmission.

## Features

### Color Management
- **Multiple Color Modes**:
  - RGB (Red, Green, Blue)
  - HSV (Hue, Saturation, Value)
  - YIQ (Luminance, In-phase, Quadrature)
  - HLS (Hue, Luminance, Saturation)
- **Pixel Order Support**:
  - RGB/GRB for WS2812
  - RGBW/GRBW for SK6812
  - Configurable pixel order to match different LED types

### Color Processing

Uses a polynomial regression gamma calculation.

- **Built-in Gamma Correction**:
  - Linear (no correction)
  - Custom gamma interpolation (see colors.py)
  - Multiple pre-defined gamma curves:
    - Default gamma (18% middle grey)
    - sRGB gamma (21.4% middle grey)
    - Simple gamma
    - Square gamma

### Performance Features
- Optimized SPI buffer encoding
- Multiple clock rates (400KHz, 800KHz, 1200KHz)
- Efficient bit-packing for reduced memory usage
- Support for up to 320 LEDs on the default spi buffer size.

### Additional Features
- Brightness control
- Auto-write mode
- Context manager support for a clean shutdown
- Custom chip select pin (BCM pin mode)
- Use operators for pixel manipulation and other effects
- A Neopixel stripe will be automatically simulated in the console on a non Rapberry Pi system
- Power control. Limit the total power consumption of the whole stripe. The brigthness will automatically reduce if neccesary. Note that the accuracy of this feature is quite limitted.

## Installation

Python >= 3.10 is required.

1. Enable SPI on your Raspberry Pi:
```bash
sudo raspi-config
# Navigate to Interface Options -> SPI -> Enable
```

2. Clone this repository:
```bash
git clone https://github.com/Hangover3832/rpi_neopixel_spi.git
```

3. Install required packages:
```bash
pip3 install -r requirements.txt
```

## Basic Usage

### Basic Examples
```python
import numpy as np
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode

# Create a strip of 60 LEDs
with RpiNeoPixelSPI(60, device=0, brightness=1.0, color_mode=ColorMode.RGB) as strip:
    # Set first LED to red
    strip[0] = (1.0, 0.0, 0.0)

    # Set second LED to green
    strip[1] = (0.0, 1.0, 0.0)

    # Set pixels 25, 30, 42 and 59 to magenta and update
    strip.set_value([25, 30, 42, 59], (1.0, 0.0, 1.0))()

    # Slicing: 
    # Set pixels 10 to 19 to blue
    strip[10:20] = np.array([0.0, 0.0, 1.0])

    # Set top half pixels to yellow and update
    strip[30:] = (1.0, 1.0, 0.0)
    strip()

    # Shift a yellow pixel in, and shift all other pixels by 1
    strip.roll(shift=1, value=(1.0, 1.0, 0.0))()
```

### Rainbow Effect With Output Power Control.

```python
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode
from time import sleep
import numpy as np

def rainbow_cycle(strip):
    # Create rainbow pattern (HSV mode)
    for i in strip:
        hue = i / strip.num_pixels
        saturation = 1.0
        value = 1.0
        strip[i] = (hue, saturation, value)

    # Rotate colors
    while True:
        strip >>= 1 # Shift all colors one pixel and update
        sleep(0.05)

# Run rainbow effect on 60 LEDs with a maximum power draw of ~5 Watts
with RpiNeoPixelSPI(60, 
        brightness=1.0, 
        color_mode=ColorMode.HSV,
        max_power=5.0, # Set the power limit to 5W
        ) as strip:

    strip.watts_per_led = np.array([0.082, 0.081, 0.079, 0.]) # Example, Watts per LED [R, G, B, W]
    rainbow_cycle(strip)
```

### Using Custom Chip Select (cs) Pin
```python
with RpiNeoPixelSPI(60, custom_cs=12) as strip:
    # use BCM pin 12 as cs
```

### Using Different Color Modes
```python
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode

with RpiNeoPixelSPI(60) as strip:
    # RGB mode
    strip.color_mode = ColorMode.RGB
    strip[0] = (1.0, 0.0, 0.0)  # Red
    
    # HSV mode (default)
    strip.color_mode = ColorMode.HSV
    strip[1] = (0.0, 1.0, 1.0)  # Red in HSV
    
    # HLS mode
    strip.color_mode = ColorMode.HLS
    strip[2] = (0.0, 0.5, 1.0)  # Red in HLS

    strip() # show strip

    # Set a YIQ value regardless of the current color mode & show()
    strip.set_value(3, (0.5, 0.5, 0.5), color_mode=ColorMode.YIQ)()

    # Using sclicing, set all pixels half brightness
    strip.color_mode = ColorMode.RGB
    strip[:] = (0.5, 0.5, 0.5)
    # and the last 10 pixels to aqua
    strip[-10:] = (0.0, 1.0, 1.0)
```

### Using operators
```python
from neopixel_spi import RpiNeoPixelSPI

with RpiNeoPixelSPI(60, gamma_func=square_gamma) as strip:
    # Make all leds white:
    strip.fill((1.0, 1.0, 1.0))

    # Dim all pixels to 75%
    strip *= 0.75

    # Add and subtract a value to each pixel
    strip += 0.6
    strip += -0.2

    # Roll all pixels to the left or right with immediate update
    strip <<= 1 # roll 1 pixel to left
    strip >>= 2 # roll 2 pixels to the right

    # Roll without immediate update
    strip >> 1 # type: ignore
    strip << 2 # type: ignore

    # Invert all colors
    ~strip # type ignore
    (~strip)() # with update

```

### Using Gamma Correction

```python
import numpy as np
from neopixel_spi import RpiNeoPixelSPI
from colors import square_gamma, create_gamma_funtion

# Using square gamma correction
with RpiNeoPixelSPI(60, gamma_func=square_gamma) as strip:
    strip.fill((0.5, 0.5, 0.5))()  # Set half brightness and show()

# Use custom gamma correction
MY_GAMMA = np.array([
    0.0, # value for 0% brithness
    0.7, # value for 50% brigthness
    1.0  # value for 100% brightness
])

my_gamma_func: Callable[[float], float] = create_gamma_function(MY_GAMMA)

with RpiNeoPixelSPI(60, gamma_func=my_gamma_func) as strip:
    strip.fill((0.5, 0.5, 0.5))()  # Set more accurate half brightness white and show()
```

This module uses a polynomial regression to apply gamma corection. See `colors.py` for more details.

## Buffer Size Limitation

By default, the driver supports up to 320 LEDs in a single buffer. For larger installations, the SPI driver needs to switch buffers, which may cause occasional glitches. To increase the buffer size, modify `/boot/firmware/cmdline.txt` by adding:

```
spidev.bufsiz=<NEEDED_BUFFER_SIZE>
```

Calculate the needed buffer size using: `number_of_pixels * 12` for RGB strips or `number_of_pixels * 16` for RGBW strips.

## Hardware Connection

Connect your NeoPixel strip to the Raspberry Pi's SPI pins:
- MOSI (GPIO 10) → Data input of NeoPixel strip
- GND → GND of NeoPixel strip
- 5V → 5V of NeoPixel strip (via appropriate power supply)

## Advanced Documentation

For detailed timing measurements and technical information about the SPI protocol implementation, see [Bit Timing Documentation](bit_timing.md).

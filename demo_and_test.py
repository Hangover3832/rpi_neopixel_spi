from time import sleep
import numpy as np
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode, PixelOrder, linear_gamma, default_gamma, create_gamma_function
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from random import random, randint


def class_test():
    with RpiNeoPixelSPI(60, color_mode=ColorMode.RGB, pixel_order=PixelOrder.GRBW, brightness=1, gamma_func=linear_gamma, max_power=10) as neo:
        neo.clear()
        neo.set_value([10, 20, 50], (1.0, 0.0, 0.0, 0.0))() # Set pixels 10, 20 and 70 to red
        neo[30:40] = 0, 0, 1, 1 # set pixels 30..39 to blue on white
        neo *= np.array([1.0, 1.0, 1.0, 0.75]) # reduce all white LEDs brithness to 75%
        neo[50] = 1.0 # This sets only the white LED on a RGBW Neopixel
        neo[50] = 1, 0, 0 # set pixel 50 to red, keep the white LED as is
        neo() # show()
        sleep(0.2)

        for _ in range(10):
            neo << 1 # left roll by 1, but not show() # type: ignore
            sleep(0.1)
            (~neo)() # invert colors and show()
            neo *= 0.9 # multiply all led values with 0.9
        for _ in range(10):
            neo += np.array([0.0, 0.0, 0.0, 0.1]) # increase all white LEDs by 0.1
            neo >>= 2 # right roll by 2 and show()
            sleep(0.1)

        neo.gamma_func = create_gamma_function(np.array([0.0, 0.75, 1.0]))

        neo.fill((1.0, 1.0, 1.0, 1.0))()
        print(f" Power at {neo.power_consumption:.1f} W")
        sleep(0.5)
        neo *= 0.5
        neo()
        print(f" Power at {neo.power_consumption:.1f} W")
        sleep(0.5)
        neo *= 0.5
        neo()
        print(f" Power at {neo.power_consumption:.1f} W")
        sleep(0.5)

        neo[:] = (1,1,1,1)
        if neo().is_simulated:
            print("full power")
        neo.clear()()


def GammaTest() -> None:
    with RpiNeoPixelSPI(100, pixel_order=PixelOrder.GRB, gamma_func=linear_gamma) as neo:
        for i in neo:
            neo[i] = 0.0, 0.0, i/(neo.num_pixels-1)
        if neo().is_simulated:
            print()
        for hue in range(0, 361, 30):
            for i in neo:
                neo[i] = hue/360, 1.0, i/(neo.num_pixels-1)
            neo()
            print(f"Hue={hue/360:.2f}")


def Rainbow():
    with RpiNeoPixelSPI(144, pixel_order=PixelOrder.GRB, gamma_func=linear_gamma) as neo:

        for i in neo:
            # Create a rainbow pattern in the default HSV space
            neo[i] = (i/(neo.num_pixels-1), 1.0, 1.0)
        while True:
            neo >>= 1 # roll the pattern and show()
            sleep(0.02)


def Raindrops():
    @Every.every(0.1)
    def drop(strip: RpiNeoPixelSPI):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, strip.num_pixels-1)
        value = random(), 1.0, 1.0 # a random color at full saturation and intensity
        strip.set_value(index, value, color_mode=ColorMode.HSV)()
        drop.interval = random()/5

    @Every.every(0.01)
    def decay(strip: RpiNeoPixelSPI):
        # reduce all pixel values
        strip += -0.005
        strip()

    with RpiNeoPixelSPI(144, max_power=2, gamma_func=default_gamma) as neo:
        while True:
            drop(neo)
            decay(neo)
            sleep(0.001)


if __name__ == "__main__":
    RpiNeoPixelSPI(320).clear()()
    GammaTest()
    #class_test()
    Raindrops()
    # Rainbow()

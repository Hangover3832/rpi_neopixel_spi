import numpy as np
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode, PixelOrder, linear_gamma, default_gamma


def class_test():
    from time import sleep

    with RpiNeoPixelSPI(100, color_mode=ColorMode.RGB, pixel_order=PixelOrder.GRBW, brightness=1, gamma_func=linear_gamma) as neo:
        neo.clear()
        neo.set_value([10, 20, 70], (1.0, 0.0, 0.0, 0.0))() # Set pixels 10, 20 and 70 to red
        neo[30:40] = 0, 0, 1, 1 # set pixels 30..39 to blue on white
        neo *= np.array([1.0, 1.0, 1.0, 0.75]) # reduce all white LEDs brithness to 75%
        neo[50] = 1.0 # This sets only the white LED on a RGBW Neopixel
        neo[50] = 1, 0, 0 # set pixel 50 to red, keep the white LED as is
        neo[60] = 0, 1, 0, 0 # set pixel 60 to green
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
        neo.fill((1.0, 1.0, 1.0, 1.0))()
        print(f" Power at {neo.power_consumption*100:.1f} %")
        neo *= 0.5
        neo()
        print(f" Power at {neo.power_consumption*100:.1f} %")
        neo *= 0.5
        neo()
        print(f" Power at {neo.power_consumption*100:.1f} %")


def GammaTest() -> None:
    with RpiNeoPixelSPI(100, device=0, color_mode=ColorMode.HSV, pixel_order=PixelOrder.GRBW) as neo:
        for i in range(neo.num_pixels):
            v = i/(neo.num_pixels-1)
            color = 1, 0, v
            neo[i] = color
        neo()


def Rainbow():
    from time import sleep
    from colors import linear_gamma
    with RpiNeoPixelSPI(144, pixel_order=PixelOrder.GRB, gamma_func=linear_gamma) as neo:
        neo.clear()() # clear() and show()
        for i in range(neo.num_pixels):
            v = i/(neo.num_pixels-1)
            color = v, 1., 1.
            neo[i] = color
        while True:
            neo >>= 1 # roll() and show()
            sleep(0.02)


def Raindrops():
    from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
    from random import random, randint
    from time import sleep

    @Every.every(0.2)
    def drop(strip: RpiNeoPixelSPI):
        # place a random colored pixel at a random location
        index = randint(0, strip.num_pixels-1)
        value = random(), 1.0, 1.0
        strip.set_value(index, value, color_mode=ColorMode.HSV)()

    @Every.every(0.01)
    def decay(strip: RpiNeoPixelSPI):
        # reduce all pixel values to 99%
        strip *= 0.99

    @Every.every(5.0)
    def print_num(num: int):
        print(f"Number of lit LEDs: {num}")

    with RpiNeoPixelSPI(100, gamma_func=linear_gamma) as neo:
        while True:
            if hasattr(neo._spi, 'IS_DUMMY_DEVICE'):
                neo._spi.message = f" {neo.num_lit_pixels} lit LEDs, Power at {neo.power_consumption*100:.1f}%"
            drop(neo)
            decay(neo)
            sleep(0.001)


if __name__ == "__main__":
    GammaTest()
    class_test()
    Raindrops()
    #Rainbow()

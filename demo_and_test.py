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
        neo[45] = 1.0 # This sets only the white LED on a RGBW Neopixel
        neo[45] = 1, 0, 0 # set pixel to red, keep the white LED as is
        neo[:10] = 0, 1, 1 # set first 10 pixels to aqua
        neo[-10:] = 1, 1, 0 # set last 10 pixels to yellow
        neo() # show()
        sleep(1)

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

    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW, gamma_func=default_gamma) as neo:
        for i in neo:
            neo[i] = 0.0, 0.0, i/(neo.num_pixels-1)
        if neo().is_simulated:
            print()


def Rainbow():

    @Every.every(0.5)
    def drop(neo: RpiNeoPixelSPI):
        """Drop in some white pixels"""
        neo[:3] = 1.0
        drop.interval = random()

    @Every.every(1)
    def gap(neo:RpiNeoPixelSPI):
        neo[:3] = 0,0,0,0
        gap.interval = random()+0.5

    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW, brightness=0.25) as neo:
        neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
        for i in neo:
            # Create a rainbow pattern in the default HSV space
            neo[i] = (i/(neo.num_pixels-1), 1.0, 1.0)
        while True:
            drop(neo)
            #gap(neo)
            neo[-1] = 0.0
            neo >>= 1 # roll the pattern and show()
            
            #decay(neo)
            # sleep(random())


def Raindrops():
    @Every.every(0.1)
    def drop(strip: RpiNeoPixelSPI):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, strip.num_pixels-1) # random position
        hue = random() # a random color at full saturation and intensity
        if random() > 0.75:
            strip[index] = 1.0
        else:
            strip.set_value(index, (hue, 1.0, 1.0), color_mode=ColorMode.HSV)()
        drop.interval = random()/5
        if strip.is_simulated:
            print()

    @Every.every(0.01)
    def decay(strip: RpiNeoPixelSPI):
        # reduce all pixel values to fade them out
        strip += -0.005
        strip()

    with RpiNeoPixelSPI(150, max_power=5, pixel_order=PixelOrder.GRBW) as neo:
        while True:
            drop(neo)
            decay(neo)
            sleep(0.001)


def power_measure():
    lin_gamma = create_gamma_function(np.array([0.0, 1.0]))
    with RpiNeoPixelSPI(10, pixel_order=PixelOrder.GRBW, color_mode=ColorMode.RGB, gamma_func=lin_gamma) as neo:
        neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
        neo[:] = 1.0, 1.0, 1.0, 1.0
        print(f"{neo().power_consumption=}")


if __name__ == "__main__":
    RpiNeoPixelSPI(320).clear()()
    GammaTest()
    class_test()
    # Raindrops()
    Rainbow()
    # power_measure()

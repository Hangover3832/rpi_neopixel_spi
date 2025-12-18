from encodings import mbcs
from time import sleep
import numpy as np
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode, PixelOrder, create_gamma_function, G
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from random import random, randint


def class_test():
    # RpiNeoPixelSPI(8, pixel_order=PixelOrder.GRB)(0, 1.0) # setting the white LED an a non RGBW throws an exception
    # RpiNeoPixelSPI(8, pixel_order=PixelOrder.GRB)[1] = 0.5 # setting the white LED an a non RGBW throws an exception

    g = lambda x: x**2.4 # Gamma 2.4
    # G.plot(g)
    RpiNeoPixelSPI(8, gamma_func=g).fill((0.5,0.5,0.5))() # Test simple lambda gamma function
    RpiNeoPixelSPI(8, gamma_func=create_gamma_function(np.array([0.1, 0.9]))).fill((0.5,0.5,0.5))() # Test custom gamma function

    with RpiNeoPixelSPI(60, color_mode=ColorMode.RGB, pixel_order=PixelOrder.GRBW, brightness=1, gamma_func=G.linear.value, max_power=10) as neo:

        print("Virtual screens:")
        screen = neo.add_virtual_screen(np.array([[5, 7, 9], [18, 19, 20]]))

        # Put some colors on the virtual screen:
        screen_data1 = np.array([
                [[1.0, 0., 0., 1.0], [0., 1., 0., 0.0], [0., 0., 1., 0.]],
                [[0., 1., 0., 0], [0., 0., 1., 0.], [1., 0., 0., 1.0]],
        ])
        if neo.virtual_screen_data(screen, screen_data1)().is_simulated:
            print()

        neo.clear()

        # Turn on some white LEDs on the virtual screen:
        screen_data2 = np.array([
                [[0.], [1.], [1.]],
                [[0.], [1.], [0.]],
        ])
        if neo.virtual_screen_data(screen, screen_data2)().is_simulated:
            print()

        neo.clear()
        neo.set_value([10, 20, 50], (1.0, 0.0, 0.0, 0.0))() # Set pixels 10, 20 and 50 to red
        neo[30:40] = 0., 0., 1., 1. # set pixels 30..39 to blue on white
        neo *= np.array([1.0, 1.0, 1.0, 0.75]) # reduce all white LEDs brithness to 75%
        neo[45] = 1.0 # This sets only the white LED on a RGBW Neopixel
        neo[45] = 1., 0., 0. # set pixel to red, keep the white LED as is
        neo[:10] = 0., 1., 1. # set first 10 pixels to aqua
        neo[-10:] = 1., 1., 0. # set last 10 pixels to yellow
        neo() # neo.show()
        sleep(1)

        i = np.array([5, 10, 15, 20])
        neo([*i], (0., 0., 0.))

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

        print("full power")
        neo[:] = (1., 1., 1., 1.)
        neo()
        neo.clear()()


def GammaTest() -> None:

    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW) as neo:
        # Create a brightness gradient
        for i in neo:
            neo[i] = 0.0, 0.0, i/(neo.num_pixels-1)
        if neo().is_simulated:
            print()


def Rainbow():

    @Every.every(0.5)
    def drop():
        """Drop in some white pixels"""
        drop.interval = random()
        
    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW, brightness=1.0) as neo:
        neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
        for i in neo:
            # Create a rainbow pattern in the default HSV space
            neo[i] = (i/(neo.num_pixels-1), 1.0, 1.0)
        while True:
            if drop()[0]:
                neo.roll(value=1.0) # drop a white pixel
            else:
                neo.roll()

            neo()[-1] = 0.0 # remove the last white pixel

            sleep(0.001)


def Raindrops():
    
    @Every.every(0.1)
    def drop(strip: RpiNeoPixelSPI):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, strip.num_pixels-1) # random position
        hue = random() # a random color in HSV color space
        if random() > 0.75:
            # Create a white pixel then and now
            strip[index] = 1.0
        else:
            strip.set_value(index, (hue, 1.0, 1.0))()
        drop.interval = random()/5

    @Every.every(0.01)
    def decay(strip: RpiNeoPixelSPI):
        # reduce all pixel values to fade them out
        strip += -0.005
        strip()

    @Every.every(0.05)
    def roll(strip: RpiNeoPixelSPI):
        strip >>= 1


    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW) as neo:
        while True:
            drop(neo)
            decay(neo)
            # roll(neo)
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
    Rainbow()
    # Raindrops()
    # power_measure()

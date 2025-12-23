from time import sleep, monotonic
from matplotlib import axis
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
        if neo.virtual_screen_data(screen, screen_data1, color_mode=ColorMode.RGB)().is_simulated:
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

        # alternative indexing
        i = np.array([5, 10, 15, 20])
        neo([*i], (0., 0., 0.))

        for _ in range(10):
            neo <<= 1 # left roll by 1, but not show() # type: ignore
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
        neo().clear()


def ColorModeTest():
    """Color mode conversion test"""
    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW) as neo:
        r = np.array([1.,0.,0.])
        g = np.array([0., 1., 0.])
        b = np.array([0. ,0., 1.])
        neo.color_mode = ColorMode.RGB
        neo.next(r)
        neo.next(g)
        neo.next(b)
        neo().color_mode = ColorMode.HSV
        v1 = neo.color_mode.from_rgb(r)
        v2 = neo.color_mode.from_rgb(g)
        v3 = neo.color_mode.from_rgb(b)
        print("R G & B in HVS", v1, v2, v3)
        neo.next(v1)
        neo.next(v2)
        neo.next(v3)
        neo().color_mode = ColorMode.HLS
        v1 = neo.color_mode.from_rgb(r)
        v2 = neo.color_mode.from_rgb(g)
        v3 = neo.color_mode.from_rgb(b)
        print("R G & B in HLS", v1, v2, v3)
        neo.next(v1)
        neo.next(v2)
        neo.next(v3)
        neo().color_mode = ColorMode.YIQ
        v1 = neo.color_mode.from_rgb(r)
        v2 = neo.color_mode.from_rgb(g)
        v3 = neo.color_mode.from_rgb(b)
        print("R G & B in YIQ", v1, v2, v3)
        neo.next(v1)
        neo.next(v2)
        neo.next(v3)
        neo().clear()


def GammaTest() -> None:

    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW) as neo:
        # Create a brightness gradient
        for i in neo:
            neo[i] = 0.0, 0.0, i/(neo.num_pixels-1)
        if neo().is_simulated:
            print()

        neo()


def Rainbow(neo: RpiNeoPixelSPI):

    @Every.every(0.5)
    def drop():
        """Drop in some white pixels"""
        drop.interval = random()

    @Every.every(5.0)
    def stopit():
        pass

    neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
    for i in neo:
        # Create a rainbow pattern in the default HSV space
        neo[i] = (i/(neo.num_pixels-1), 1.0, 1.0)

    while True:
        if drop()[0]:
            neo.roll(value=1.0) # drop a white pixel
        else:
            neo.roll()

        neo()[-1] = 0.0 # remove the last white pixel so it doesn't roll in again
        sleep(0.001)

        if stopit()[0]:
            break


def Raindrops(neo: RpiNeoPixelSPI):
    neo.gamma_func = G.linear.value
    
    @Every.every(0.1)
    def drop(strip: RpiNeoPixelSPI):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, strip.num_pixels-1) # random position
        hue = random() # a random color in HSV color space
        strip.set_value(index, (hue, 1.0, 1.0))
        drop.interval = random()/5

    @Every.every(1.0)
    def dropW(strip: RpiNeoPixelSPI):
        # place a white pixel at a random location every second
        index = randint(0, strip.num_pixels-1) # random position
        value = random() # a random color in HSV color space
        strip.set_value(index, value)


    @Every.every(0.05)
    def roll(neo):
        neo <<= 1

    @Every.every(30.0)
    def stopit():
        pass

    while True:
        drop(neo)
        dropW(neo)
        neo *= 0.98 # pixel decay
        neo()
        # roll(neo)
        sleep(0.001)
        if stopit()[0]:
            break


def power_measure():
    lin_gamma = create_gamma_function(np.array([0.0, 1.0]))
    with RpiNeoPixelSPI(10, pixel_order=PixelOrder.GRBW, color_mode=ColorMode.RGB, gamma_func=lin_gamma) as neo:
        neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
        neo[:] = 1.0, 1.0, 1.0, 1.0
        neo()
        print(f"{neo().power_consumption=}")


def light_show():
    with RpiNeoPixelSPI(150, pixel_order=PixelOrder.GRBW, max_power=0.5) as neo:
        while True:
            Rainbow(neo)
            Raindrops(neo)


def fire():
    from effects import Fire

    candle = Fire(
        RpiNeoPixelSPI(24, device=0, pixel_order=PixelOrder.GRB, brightness=1.0),
        spectrum=(0.5, -0.2),
        decay_factor=(0.95, 0.9),
        spark_interval_factor=0.05,
        spark_propagation_delay=0.01
        )

    candle.neo.reversed = False
    candle.ignite_spark.pause()
    candle._temperature_gradient()

    t = monotonic() + 2.5
    while monotonic() < t:
        candle.progress()

    candle.ignite_spark.resume()
    while True:
        candle.progress()


if __name__ == "__main__":
    RpiNeoPixelSPI(320, device=0).clear()()
    RpiNeoPixelSPI(320, device=1).clear()()
    #GammaTest()
    #class_test()
    #ColorModeTest()
    #power_measure()
    #light_show()
    fire()

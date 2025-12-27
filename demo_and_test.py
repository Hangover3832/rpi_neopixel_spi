from time import sleep, monotonic
from matplotlib import axis
import numpy as np
from neopixel_spi import RpiNeoPixelSPI
from colors import ColorMode, PixelOrder, create_gamma_function, G, rgb_to_hsv, hsv_to_rgb, SOME_COLORS
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from random import random, randint
from effects import Fire, Meteor

def class_test():
    # RpiNeoPixelSPI(8, pixel_order=PixelOrder.GRB)(0, 1.0) # setting the white LED an a non RGBW throws an exception
    # RpiNeoPixelSPI(8, pixel_order=PixelOrder.GRB)[1] = 0.5 # setting the white LED an a non RGBW throws an exception

    g = lambda x: x**2.4 # Gamma 2.4
    # G.plot(g)
    RpiNeoPixelSPI(8, gamma_func=g).fill((0.5,0.5,0.5))() # Test simple lambda gamma function
    RpiNeoPixelSPI(8, gamma_func=create_gamma_function(np.array([0.1, 0.9]))).fill((0.5,0.5,0.5))() # Test custom gamma function

    with RpiNeoPixelSPI(60, color_mode=ColorMode.RGB, pixel_order=PixelOrder.GRBW, brightness=1, gamma_func=G.linear.value, max_power=10) as neo:

        print("Virtual screens")
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
        neo([10, 20, 50], (1.0, 0.0, 0.0, 0.0))() # Set pixels 10, 20 and 50 to red
        neo[30:40] = 0., 0., 1., 1. # set pixels 30..39 to blue on white
        neo *= np.array([1.0, 1.0, 1.0, 0.75]) # reduce all white LEDs brithness to 75%
        neo[45] = 1.0 # This sets only the white LED on a RGBW Neopixel
        neo[45] = 1., 0., 0. # set pixel to red, keep the white LED as is
        neo[:10] = 0., 1., 1. # set first 10 pixels to aqua
        neo[-10:] = 1., 1., 0. # set last 10 pixels to yellow
        neo() # neo.show()

        # alternative indexing
        i = np.array([5, 10, 15, 20])
        neo([*i], (0., 0., 0.))

        for _ in range(10):
            neo <<= 1 # left roll by 1, but not show() # type: ignore
            (~neo)() # invert colors and show()
            neo *= 0.9 # multiply all led values with 0.9
        for _ in range(10):
            neo += np.array([0.0, 0.0, 0.0, 0.1]) # increase all white LEDs by 0.1
            neo >>= 2 # right roll by 2
            neo()

        neo.clear()()


    with RpiNeoPixelSPI(10, device=1, pixel_order=PixelOrder.GRBW) as neo:
        
        neo.clear()
        neo[:] = 0.1 # add white pixel values

        def read_and_write(cm: ColorMode):
            print()
            neo.color_mode = cm
            v = (0.9, 0.45, 0.18)
            print(f"Write value {v} in {cm.name}")
            neo[0] = v
            neo(1, v)
            print(f"Read value: {neo[0]}")

            print(f"Write values in {cm.name}:")
            for i in range(neo.num_pixels):
                v = i/neo.num_pixels, i/neo.num_pixels/2, i/neo.num_pixels/5
                print(v)
                neo[i] = v
            
            print(f"Read sliced values:\n {neo[:]}")

            print()
            neo()

        for cm in ColorMode:
            read_and_write(cm)


        neo.color_mode = ColorMode.HSV
        n = 5
        v = np.array([[0.0, 1.0, 1.0]]) 
        for i in range(1, n):
            v = np.vstack((v, np.array([[i/n, 1.0, 1.0]])))

        print(f"Array broadcast HSV rainbow:\n{v}")
        neo[4] = v # broadcast somwhere
        neo[0] = v # broadcast to index 0
        neo[-1:-n-1:-1] = v # reversed broadcast from the end
        # neo *= 0.1 # dim the stripe
        neo()
        # print all values HSV:
        print(neo[:])

        neo.clear()()


def ColorModeTest():
    print("\nColor mode conversion tests:")

    def run_test():

        def process_mode(mode: ColorMode):
            print(f"Mode {mode.name}:")

            for i, (name, color) in enumerate(SOME_COLORS.items()):
                color = color[:3]
                v1 = ColorMode.RGB.convert_to(color, mode)
                v2 = mode.convert_to(v1, ColorMode.RGB)
                print(f"[{i}] RGB {color} to {mode.name}: {np.round(v1, 3)}")
                print(f"[{i}] {mode.name}->RGB->{mode.name}:         {np.round(v2, 3)}")

        for cm in ColorMode:
            process_mode(cm)

    run_test()


def GammaTest() -> None:

    with RpiNeoPixelSPI(150, device=1, pixel_order=PixelOrder.GRBW) as neo:
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

    neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
    for i in neo:
        # Create a rainbow pattern in the default HSV space
        neo[i] = (i/(neo.num_pixels-1), 1.0, 1.0)


    @Every.While(5, n=neo) # repeat for 5s
    def proceed(n:RpiNeoPixelSPI):
        if drop():
            n.roll(value=1.0) # drop a white pixel
        else:
            n.roll()
        n()[-1] = 0.0 # remove the last white pixel so it doesn't roll in again


def Raindrops(neo: RpiNeoPixelSPI):
    neo.gamma_func = G.linear.value
    
    @Every.every(0.1)
    def drop(n: RpiNeoPixelSPI):
        # place a random colored pixel at a random location in a random interval
        index = randint(0, n.num_pixels-1) # random position
        hue = random() # a random color in HSV color space
        n(index, (hue, 1.0, 1.0))
        drop.interval = random()/5

    @Every.every(1.0)
    def dropW(n:RpiNeoPixelSPI):
        # place a white pixel at a random location every second
        index = randint(0, n.num_pixels-1) # random position
        value = random() # a random color in HSV color space
        n(index, value)

    @Every.While(30, n=neo) # repeat for 30s
    def proceed(n: RpiNeoPixelSPI):
        drop(n)
        dropW(n)
        n *= 0.98 # pixel decay
        n()


def light_show():
    with RpiNeoPixelSPI(150, device=1, pixel_order=PixelOrder.GRBW) as neo:
        neo.reversed = True
        while True:
            Rainbow(neo)
            Raindrops(neo)


def power_measure():
    lin_gamma = create_gamma_function(np.array([0.0, 1.0]))
    with RpiNeoPixelSPI(100, device=1, pixel_order=PixelOrder.GRBW, color_mode=ColorMode.RGB, gamma_func=lin_gamma) as neo:
        neo.watts_per_led = np.array([0.042, 0.042, 0.042, 0.084])
        neo[:] = 1.0, 1.0, 1.0, 1.0
        neo().clear()()
        print(f"{neo().power_consumption=}")


def fire():
    candle1 = Fire(
        RpiNeoPixelSPI(23, device=0, pixel_order=PixelOrder.GRB, brightness=0.25),
        spectrum=(0.8, 0.0),
        decay_factor=(0.95, 0.85),
        spark_interval_factor=0.05,
        spark_propagation_interval=0.01
        )

    candle2 = Fire(
        RpiNeoPixelSPI(23, device=1, pixel_order=PixelOrder.GRBW, brightness=1.0),
        #spectrum=(1.0, 0.0),
        #decay_factor=(0.95, 0.85),
        #spark_interval_factor=0.15,
        #spark_propagation_interval=0.01
        )

    # Let the candles burn:
    while True:
        candle1.progress()
        #candle2.progress()


def meteor_shower():
    meteor = Meteor(RpiNeoPixelSPI(23, device=0, pixel_order=PixelOrder.GRB, brightness=1.0))
    while True:
        meteor.progress()


if __name__ == "__main__":
    RpiNeoPixelSPI(23, device=0, pixel_order=PixelOrder.GRB).clear()()
    RpiNeoPixelSPI(150, device=1, pixel_order=PixelOrder.GRBW).clear()()
    #GammaTest()
    class_test()
    ColorModeTest()
    #power_measure()
    #light_show()
    fire()
    meteor_shower()

from random import random, randint
from time import sleep
import numpy as np
from pyparsing import col
from colors import ColorMode, TempColor, create_gamma_function
from neopixel_spi import RpiNeoPixelSPI
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py
from typing import Tuple


class Fire:
    """
    Fire effect for the neopixel_spi library.

    :param neopixel: A Neopixel instance
    :type neopixel: RpiNeoPixelSPI
    :param spectrum: Lower and upper bound of the color temperature (black body radiation spectrum).
        A values near 0.0 is redish, near 1.0 is blueish.
        Values beyond 0..1 are possible to narrow the spectrum down.
    :type spectrum: tuple[lower bound:float, upper bound:float]
    :param decay_factor: Determines how fast the flame decays at the bottom (1st value)
        and at the top (2nd value). A lower value leads to a faster decay.
    :type decay_factor: tuple[bottom:float, top:float]
    :param spark_interval_factor: Delay between flame spark ignitions near the bottom.
        A lower value leads to more spark ignites.
    :type spark_interval_factor: float
    :param spark_propagation_delay: How fast a ignited sparl traveles up the flame.
        A lower value means faster.
    :type spark_propagation_delay: float
    """
    def __init__(self, 
                 neopixel:RpiNeoPixelSPI,
                 spectrum:Tuple[float, float]=(0.5, -0.2),
                 decay_factor:Tuple[float, float]=(0.95, 0.85),
                 spark_interval_factor:float=0.05,
                 spark_propagation_delay:float=0.01,) -> None:

        self.neo = neopixel
        self.spectrum = spectrum
        self.decay_factor = decay_factor
        self.spark_interval_factor = spark_interval_factor
        self.spark_propagation_delay = spark_propagation_delay
        neopixel.auto_write = True

    @property
    def decay_factor(self) -> Tuple[float, float]:
        return self._decay_factor

    @decay_factor.setter
    def decay_factor(self, value: Tuple[float, float]) -> None:
        self._decay_factor = value
        self._decay_array = np.linspace(value[0], value[1], self.neo.num_pixels)[:, np.newaxis]


    def get_indexed_temp(self, index:int) -> float:
        """Return the color temperature based on the pixel index.
        We use only a part of the low temperature spectrum, 
        at the bottom hotter (more blueish) at the top colder (more redish)
        based on the self.spectrum parameter"""
        x = index / (self.neo.num_pixels-1)
        result = float(np.interp(x, (0.0, 1.0), self.spectrum))
        return result


    @Every.every(1.0/30) # 30fps decay
    def decay(self) -> None:
        self.neo *= self._decay_array


    @Every.every(0.1)
    def ignite_spark(self):
        start = randint(0, self.neo.num_pixels // 4) # start at the lower 4rd
        end = randint(self.neo.num_pixels - self.neo.num_pixels // 4, self.neo.num_pixels) # end at the upper 4rd
        for i in range(start, end):
            self.neo.set_temperature(i, self.get_indexed_temp(i))
            sleep(self.spark_propagation_delay)
            self.decay(self) # timed decay function
        # add randomness to the interval
        self.ignite_spark.interval = self.spark_interval_factor*random()+self.spark_interval_factor

    def show_temperature_gradient(self):
        for i in self.neo:
            self.neo.set_temperature(i, self.get_indexed_temp(i))


    def progress(self):
        self.ignite_spark(self)
        self.decay(self)

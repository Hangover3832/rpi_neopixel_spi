from random import random, randint
from time import sleep
import numpy as np
from pyparsing import col
from colors import ColorMode, TempColor, create_gamma_function
from neopixel_spi import RpiNeoPixelSPI
from every import Every # https://raw.githubusercontent.com/Hangover3832/every_timer/refs/heads/main/every.py


class Fire:
    def __init__(self, 
                 neopixel:RpiNeoPixelSPI, 
                 spectrum:tuple[float, float]=(0.5, -0.2),
                 decay_factor:tuple[float, float]=(0.95, 0.85),
                 spark_interval_factor:float=0.05,
                 spark_propagation_delay:float=0.01,) -> None:
        
        self.neo: RpiNeoPixelSPI = neopixel
        self.spectrum: tuple[float, float] = spectrum
        self._decay_factor = decay_factor
        self.spark_interval_factor = spark_interval_factor
        self.spark_propagation_delay = spark_propagation_delay
        neopixel.auto_write = True
        self.decay_factor = decay_factor

    @property
    def decay_factor(self) -> tuple[float, float]:
        return self._decay_factor

    @decay_factor.setter
    def decay_factor(self, value: tuple[float, float]) -> None:
        self._decay_factor = value
        self._decay_array = np.linspace(value[0], value[1], self.neo.num_pixels)

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
        self.neo *= self._decay_array[:, None]

    @Every.every(0.1)
    def ignite_spark(self):
        start = randint(0, self.neo.num_pixels // 3)
        end = randint(self.neo.num_pixels - self.neo.num_pixels // 4, self.neo.num_pixels)
        for i in range(start, end):
            self.neo.set_temperature(i, self.get_indexed_temp(i))
            sleep(self.spark_propagation_delay)
            self.decay(self)
        self.ignite_spark.interval = self.spark_interval_factor*random()+self.spark_interval_factor

    def _temperature_gradient(self):
        for i in self.neo:
            self.neo.set_temperature(i, self.get_indexed_temp(i))


    def progress(self):
        self.ignite_spark(self)
        self.decay(self)

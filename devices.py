from abc import ABC
import numpy as np
from colors import PixelOrder
from enum import Enum


class Spi_Clock(Enum): # SPI clock rates
    CLOCK_400KHZ  = 1_625_000
    CLOCK_800KHZ  = 3_250_000
    CLOCK_1200KHZ = 6_500_000


try:
    from spidev import SpiDev # type: ignore
    from gpiozero import OutputDevice  # type: ignore
except:
    print("""
            Note: The python libraries 'gpiozero' and/or 'spidev' could not be imported.
            Using dummy devices on a non Rapberry PI system to simulate a Neopixel stripe in the console.
          """)


    class DummyRPIDev(ABC):

        IS_DUMMY_DEVICE = True

        def __init__(self) -> None:
            print(f"Initializig dummy device '{self.__class__.__name__}'", end="")

        def open(self) -> None:
            print(f"Dummy device open '{self.__class__.__name__}'")

        def close(self) -> None:
            print(f"Dummy device close '{self.__class__.__name__}'")

        def on(self) -> None:
            return
        
        def off(self) -> None:
            return


    class SpiDev(DummyRPIDev):
        """
        We simulate a GRB(W) neopixel, decode the provided spi byte stream
        and print a simulated Neopixel stripe
        """

        LED_CHAR = "\u25CF"

        def __init__(self, pixel_order: PixelOrder) -> None:
            super().__init__()
            print()
            self.max_speed_hz: int = 0
            self.mode = 0
            self.bits_per_word = 8
            self.no_cs = False
            self.pixel_order = pixel_order

        def open(self, bus: int = 0, device: int = 0) -> None:
            print(f"Dummy SPI open, {bus=}, {device=}")

        def writebytes2(self, buffer: np.ndarray) -> None:

            def convert(bits: np.ndarray) -> int:
                # bits = np.ndarray[uint8, uitn8, uint8, uint8] = 4 double bits = 8 bits = 1 byte
                bit_values = {0xCC: 0b11, 0xC8: 0b10, 0x8C: 0b01, 0x88: 0b00} # SPI encodings {1byte: 2bits}
                result = 0
                for bit in bits:
                    result = (result << 2) | bit_values[bit] # shift 2 bits and inject 2 new bits
                return result

            double_bits_per_pixel = 12 if len(self.pixel_order.name) == 3 else 16 # check if is GRB or GRBW...
            buffer = buffer.reshape([buffer.shape[0]//double_bits_per_pixel, double_bits_per_pixel]) # ...and reshape the buffer accordingly
            print('', end='\r')
            for bits in buffer: 
                g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
                w = convert(bits[12:16]) if bits.shape[0]>12 else 0
                print(f"\033[48;2;{w};{w};{w}m", end='') # the background color simulates the white LED in a GRBW Neopixel
                print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LEDs


    class OutputDevice(DummyRPIDev):
        """Dummy class for a custom chip select pin"""
        def __init__(self, pin: int,  active_high: bool, initial_value: bool) -> None:
            super().__init__()
            print(f", {pin=}")

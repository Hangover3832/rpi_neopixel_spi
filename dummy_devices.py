from abc import ABC
import numpy as np


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

    LED_CHAR = "\u25CF"

    def __init__(self) -> None:
        super().__init__()
        print()
        self.max_speed_hz: int = 0
        self.mode = 0
        self.bits_per_word = 8
        self.no_cs = False

    def open(self, bus: int = 0, device: int = 0) -> None:
        print(f"Dummy SPI open, {bus=}, {device=}")

    def writebytes2(self, buffer: np.ndarray) -> None:
        # We simulate a GRB(W) neopixel, decode the provided spi byte stream

        def convert(bits: np.ndarray) -> int:
            # bits = np.ndarray[uint8, uitn8, uint8, uint8] = 4 double bits = 8 bits = 1 byte
            bit_values = {0xCC: 0b11, 0xC8: 0b10, 0x8C: 0b01, 0x88: 0b00} # SPI encodings {1byte: 2bits}
            result = 0
            for bit in bits:
                result = (result << 2) ^ bit_values[bit] # shift 2 bits and inject 2 new bits
            return result

        # buffer = buffer.reshape([buffer.shape[0]//12, 12])
        print('', end='\r')
        for bits in buffer: 
            g, r, b = convert(bits[0:4]), convert(bits[4:8]), convert(bits[8:12])
            w = convert(bits[12:16]) if bits.shape[0]>12 else 0
            print(f"\033[48;2;{w};{w};{w}m", end='')
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='', flush=True) # print the LED's


class OutputDevice(DummyRPIDev):
    def __init__(self, pin: int,  active_high: bool, initial_value: bool) -> None:
        super().__init__()
        print(f", {pin=}")

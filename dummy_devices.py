from abc import ABC
import numpy as np
from functools import reduce

class DummyRPIDev(ABC):

    IS_DUMMY_DEVICE = True

    def __init__(self) -> None:
        print(f"Initializig dummy device '{self.__class__.__name__}'", end="")

    def open(self):
        print("Dummy device open")

    def close(self):
        print("Dummy device close")

    def on(self):
        return
    
    def off(self):
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

    def open(self, bus: int = 0, device: int = 0):
        print(f"Dummy SPI open, {bus=}, {device=}")

    def writebytes2(self, buffer: np.ndarray):
        # We simulate a GRB neopixel and completely decode the provided bit stream

        def convert(bits: np.ndarray) -> int:
            # bits = np.ndarray[uint8, uitn8, uint8, uint8]
            result = 0
            for bit in bits:
                result = result << 2
                match bit:
                    case 0xCC:
                        result ^= 0b11
                    case 0xC8:
                        result ^= 0b10
                    case 0x8C:
                        result ^= 0b01
                    case 0x88:
                        result ^= 0b00
                    case other:
                        raise Exception(f"Invalid bit pattern: {hex(other)}")
            return result

        # buffer = buffer.reshape([buffer.shape[0]//12, 12])
        print('', end='\r')
        for bits in buffer:
            g = convert(bits[:4])
            r = convert(bits[4:8])
            b = convert(bits[8:])
            print(f"\033[38;2;{r};{g};{b}m{self.LED_CHAR}\033[0m", end='')


class OutputDevice(DummyRPIDev):
    def __init__(self, pin: int,  active_high: bool, initial_value: bool) -> None:
        super().__init__()
        print(f", {pin=}")

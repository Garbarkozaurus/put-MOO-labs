import numpy as np
from typing import Iterable


class Company:
    def __init__(self, name: str, prices: Iterable[float]) -> None:
        self.name = name
        self.prices = np.array(prices)
        self.expected_return = None

    def __str__(self) -> str:
        return self.name + "\n" + str(self.prices)

import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable


def plot_SBX_distribution(
        weight1: float, weight2: float, distr_index: int,
        num_runs: int) -> None:
    first = []
    second = []
    for _ in range(num_runs):
        u = np.random.rand()
        beta = None
        if u <= 0.5:
            beta = (2 * u) ** (1/(distr_index+1))
        else:
            beta = (1/(2*(1-u))) ** (1/(distr_index+1))
        x1 = 0.5 * ((1+beta) * weight1 + (1-beta) * weight2)
        x2 = 0.5 * ((1-beta) * weight1 + (1+beta) * weight2)
        first.append(x1)
        second.append(x2)
    plt.hist(first, color="blue")
    plt.hist(second, color="red")
    plt.xlim(0, 1)
    plt.show()


def SBX_beta(distr_index: int) -> float:
    random_number = np.random.rand()
    if random_number <= 0.5:
        return (2 * random_number) ** (1/(distr_index+1))
    else:
        return (1/(2*(1-random_number))) ** (1/(distr_index+1))


def SBX_portfolios(weights1: Iterable[float], weights2: Iterable[float],
                   distr_index: int
                   ) -> tuple[Iterable[float], Iterable[float]]:
    # How should this actually work?
    # - Crossover each weight independently
    # - The offspring are just a combination of the two parents
    beta = SBX_beta(distr_index)
    offspring1 = np.array([0.5 * ((1+beta) * x1 + (1-beta) * x2) for x1, x2 in zip(weights1, weights2)])
    offspring2 = np.array([0.5 * ((1-beta) * x1 + (1+beta) * x2) for x1, x2 in zip(weights1, weights2)])
    offspring1 = np.clip(offspring1, 0, None)
    offspring2 = np.clip(offspring2, 0, None)
    offspring1 = offspring1 / np.sum(offspring1)
    offspring2 = offspring2 / np.sum(offspring2)
    return offspring1, offspring2


if __name__ == "__main__":
    plot_SBX_distribution(0.2, 0.8, 10, 1000)
    PORTFOLIO1 = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0]
    PORTFOLIO2 = [0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]

    print(SBX_portfolios(PORTFOLIO1, PORTFOLIO2, 10))

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


def mutate_portfolio(weights: Iterable[float],
                     weight_change_probability: float = 0.05,
                     s_deviation: float = 0.1) -> None:
    """The operation is performed in place!"""
    for value in weights:
        if np.random.random() < weight_change_probability:
            value += np.random.normal(0, s_deviation)
            value = np.clip(value, 0, None)
    s = np.sum(weights)
    for value in weights:
        value /= s


def random_portfolio_population(
        num_variables: int, population_size: int) -> np.ndarray[np.float32]:
    members = np.random.random(size=(population_size, num_variables))
    for row in members:
        row /= np.sum(row)
    return members


def export_population(
        population: np.ndarray[np.float32],
        file_path: str, parameters: dict,
        generation_number: int,
        mode: str, skip_header: bool = False) -> None:
    with open(file_path, mode) as fp:
        if not skip_header:
            fp.write(",".join(map(lambda x: f"{x[0]},{x[1]}", parameters.items())))
            fp.write("\n")
        for individual in population:
            fp.write(f"{generation_number};{','.join(map(str, individual))}\n")


def export_population_points(points: np.ndarray[np.float32], file_path: str, parameters: dict, generation_numbers: list[int]) -> None:
    with open(file_path, "w+") as fp:
        fp.write(",".join(map(lambda x: f"{x[0]},{x[1]}", parameters.items())))
        fp.write("\n")
        for gen, (x, y) in zip(generation_numbers, points):
            fp.write(f"{gen};{x},{y}\n")


def load_population_points(file_path: str) -> tuple[dict, list[int], np.ndarray[np.float32]]:
    def process_line(line: str) -> tuple[int, float, float]:
        g_str, point_str = line.split(';')
        gen = int(g_str)
        ret, risk = point_str.split(',')
        ret = np.float32(ret)
        risk = np.float32(risk)
        return gen, ret, risk

    with open(file_path, "r") as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
        params_line = lines[0].split(',')
        parameters = dict(zip(params_line[::2], params_line[1::2]))
        parameters["n_objectives"] = int(parameters["n_objectives"])
        parameters["neighborhood_size"] = int(parameters["neighborhood_size"])
        parameters["generations"] = int(parameters["generations"])
        parameters["population_size"] = int(parameters["population_size"])
        parameters["crossover_distr_idx"] = int(parameters["crossover_distr_idx"])
        parameters["mutation_probability"] = float(parameters["mutation_probability"])
        gens = []
        points = []
        for line in lines[1:]:
            g, ret, risk = process_line(line)
            gens.append(g)
            points.append((ret, risk))
    return parameters, gens, np.array(points, dtype=np.float32)


def plot_experiment_points(parameters: dict, generations: list[int], points: np.ndarray[np.float32]):
    # for start in range(0, len(generations)-parameters["population_size"], parameters["population_size"]):
    #     end = start + parameters["population_size"]
    #     plt.plot(points[start:end, 0], points[start:end, 1], "o", label=generations[start], alpha=0.5)
    # plt.legend(title="Generation", loc=2)
    # plt.gcf().set_size_inches(10, 10)
    # plt.show()
    unique_generations = np.unique(generations)
    avg_dict_ret = dict(zip(unique_generations, np.zeros((len(unique_generations), parameters["population_size"]), dtype=np.float32)))
    avg_dict_risk = dict(zip(unique_generations, np.zeros((len(unique_generations), parameters["population_size"]), dtype=np.float32)))
    counting_dict = dict.fromkeys(unique_generations, 0)
    for i, gen in enumerate(generations):
        avg_dict_ret[gen][i % parameters["population_size"]] += points[i][0]
        avg_dict_risk[gen][i % parameters["population_size"]] += points[i][1]
        counting_dict[gen] += 1
    for g in unique_generations:
        avg_dict_ret[g] /= counting_dict[g]
        avg_dict_risk[g] /= counting_dict[g]
        avg_dict_ret[g] *= parameters["population_size"]
        avg_dict_risk[g] *= parameters["population_size"]

    for g in unique_generations:
        plt.plot(avg_dict_ret[g], avg_dict_risk[g], "o", label=g, alpha=0.5)
    plt.legend(title="Generation", loc=2)
    plt.grid()
    plt.gcf().set_size_inches(10, 10)
    plt.show()


def load_population(file_path: str) -> tuple[dict, np.ndarray[np.float32], np.ndarray[np.int32]]:
    """Returns:
    - parameter dictionary
    - individuals (portfolios) in a single numpy matrix
    - np.ndarray with the generations when each individual was generated
    """
    def process_line(line: str) -> list[tuple[int, list[np.float32]]]:
        generation, weights = line.split(';')
        generation = int(generation)
        weights = list(map(np.float32, weights.split(',')))
        return np.array(generation), np.array(weights)

    with open(file_path, "r") as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
        params_line = lines[0].split(',')
        parameters = dict(zip(params_line[::2], params_line[1::2]))
        parameters["n_objectives"] = int(parameters["n_objectives"])
        parameters["neighborhood_size"] = int(parameters["neighborhood_size"])
        parameters["generations"] = int(parameters["generations"])
        parameters["population_size"] = int(parameters["population_size"])
        parameters["crossover_distr_idx"] = int(parameters["crossover_distr_idx"])
        parameters["mutation_probability"] = float(parameters["mutation_probability"])

        generations, population = zip(*[process_line(line) for line in lines[1:]])
    population = np.array(population)
    generations = np.array(generations)
    return parameters, population, generations


def inverted_generational_distance(
        front_coordinates: list[tuple[float, float]],
        population_coordinates: list[tuple[float, float]],
        exponent: int = 2) -> float:
    def point_distance(
            ref_point: tuple[float, float],
            portfolio_point: tuple[float, float]) -> float:
        return np.sqrt(sum([(a-b)**2 for a, b in zip(ref_point, portfolio_point)]))
    min_distances = []
    for reference_point in front_coordinates:
        distances = [point_distance(reference_point, p_point) for p_point in population_coordinates]
        min_dist = np.min(distances)
        min_distances.append(min_dist)
    dist_sum = np.sum(np.power(min_distances, exponent))
    return np.power(dist_sum, 1/exponent) / len(front_coordinates)


def num_unique_individuals_in_pop(population: np.ndarray[np.float32]) -> int:
    return len(np.unique(population, axis=0))


if __name__ == "__main__":
    plot_SBX_distribution(0.2, 0.8, 10, 1000)
    PORTFOLIO1 = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0]
    PORTFOLIO2 = [0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]

    print(SBX_portfolios(PORTFOLIO1, PORTFOLIO2, 10))
    params, gens, points = load_population_points("populations\EXPERIMENT_2024-03-16-22-25-19.txt")
    plot_experiment_points(params, gens, points)

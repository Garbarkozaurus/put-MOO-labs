import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from company import Company
from data_loading import load_saved_front
import evolutionary_operators
import utils


def plot_experiment_points(parameters: dict, generations: list[int], points: np.ndarray[np.float32]):
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


def plot_population(
        companies: list[Company],
        population: np.ndarray[np.float32],
        generation_rel: float,
        history_len: int | None = None,
        title: str = "",
        export_pdf: bool = False,
        pdf_title: str = "pop1.pdf",
        show: bool = True, alpha: float = 1.0,
        force_color: str | None = None) -> None:
    plot_points = []
    returns = []
    risks = []
    label_font = {'fontname': 'Times New Roman'}
    for sol in population:
        exp_ret = utils.portfolio_expected_return(companies, sol)
        risk = utils.portfolio_risk(companies, sol, history_len)
        plot_points.append((exp_ret, risk))
        returns.append(exp_ret)
        risks.append(risk)

    if not force_color:
        plt.plot(returns, risks, "o", alpha=alpha, c=cm.viridis(generation_rel))
    else:
        plt.plot(returns, risks, "o", alpha=alpha, c=force_color)
    plt.xlabel("Expected return [100%]", **label_font)
    plt.ylabel("Risk [$ \$^2 $] ", **label_font)
    plt.grid()
    plt.title(title, **label_font)
    if export_pdf:
        plt.savefig(pdf_title, format="pdf")
    if show:
        plt.show()


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


def plot_convergence_inverted_gen_distance(
        front_points: np.ndarray[np.float32],
        parameters: dict, generations: list[int],
        points: np.ndarray[np.float32], color: str, show: bool=True,
        save_pdf: bool = False):
    unique_generations = np.unique(generations)
    unique_generations = np.sort(unique_generations)
    distances = [[] for _ in unique_generations]
    averages = []
    minima = []
    maxima = []
    for i, g in enumerate(unique_generations):
        indices = np.where(generations==g)
        selected_members = points[indices]
        selected_members = np.reshape(selected_members, (-1, parameters["population_size"], 2))
        for pop in selected_members:
            distances[i].append(inverted_generational_distance(front_points, pop))
        averages.append(np.mean(distances[i]))
        minima.append(np.min(distances[i]))
        maxima.append(np.max(distances[i]))
    averages = np.array(averages)
    minima = np.array(minima)
    maxima = np.array(maxima)
    plt.plot(unique_generations, averages, c=color)
    plt.fill_between(unique_generations, averages-minima, averages+maxima, alpha=0.5, facecolor=color)
    plt.xticks(unique_generations[::5])
    title_font = {"fontname": "Times New Roman"}
    title = f"IGD pop_size={parameters['population_size']}, gen={parameters['generations']}"
    plt.title(title, **title_font)
    plt.gcf().set_size_inches(8.8, 5.4)
    if save_pdf:
        plt.savefig(title, format="pdf")
    if show:
        plt.show()


def export_igd_from_file(
        front_points: np.ndarray[np.float32],
        point_path: str, export_path: str) -> None:
    with open(export_path, "+a") as fp:
        parameters, generations, points = evolutionary_operators.load_population_points(point_path)
        max_gen = np.max(generations)
        indices = np.where(generations == max_gen)
        selected_members = points[indices]
        selected_members = np.reshape(selected_members, (-1, parameters["population_size"], 2))
        distances = [0 for _ in range(len(selected_members))]
        for i, pop in enumerate(selected_members):
            distances[i] = inverted_generational_distance(front_points, pop)
        param_info = f"{parameters['generations']};{parameters['population_size']}"
        point_str = ",".join(map(str, distances))
        fp.write(f"{param_info};{point_str}\n")


def igd_heatmap_from_file(igd_path: str) -> None:
    fp = open(igd_path, "r")
    lines = fp.readlines()
    fp.close()
    side_len = int(np.sqrt(len(lines)))
    means = []
    sds = []
    generations = []
    pop_sizes = []
    grid = np.zeros((side_len, side_len))
    for line in lines:
        g, p, igds = line.split(';')
        generations.append(int(g))
        pop_sizes.append(int(p))
        igds = igds[:-1].split(',')
        igds = np.array(list(map(np.float32, igds)))
        means.append(np.mean(igds))
        sds.append(np.std(igds))
    s_gens = np.sort(np.unique(generations))
    s_pops = np.sort(np.unique(pop_sizes))
    for i, mean in enumerate(means):
        row = np.where(s_pops == pop_sizes[i])[0][0]
        column = np.where(s_gens == generations[i])[0][0]
        grid[row][column] = mean
        text = f"{mean:.5f}\n({sds[i]:.5f})"
        plt.text(row, column, text, ha="center", va="center", weight="bold", c="white")
    plt.imshow(grid, cmap="viridis_r")
    plt.yticks(list(range(side_len)), labels=s_pops)
    plt.xticks(list(range(side_len)), labels=s_gens)
    plt.xlabel("Generations")
    plt.ylabel("Population size")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    POINT_PATH = "populations/EXPERIMENT_2024-03-22-21-29-09.txt"
    ec_front = load_saved_front("./saved_fronts/ec_front1.txt")
    # parameters, generations, points = evolutionary_operators.load_population_points(POINT_PATH)
    # plot_experiment_points(parameters, generations, points)
    # plot_convergence_inverted_gen_distance(ec_front, parameters, generations, points, "green", save_pdf=True)
    # files = Path("./populations/").glob("EXPERIMENT*.txt")
    # for file in files:
    #     export_igd_from_file(ec_front, file, "igd_points.txt")
    igd_heatmap_from_file("igd_points.txt")

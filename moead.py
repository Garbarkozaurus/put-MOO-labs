import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from company import Company
import data_loading
import return_estimation
import evolutionary_operators
import utils
from typing import Iterable, Callable


def evaluate_portfolio_chebyshev(
        companies: list[Company],
        portfolio_weights: Iterable[float],
        objective_weights: tuple[float]) -> float:
    """Implicit objective order: expected_return, risk, number of included
    companies"""
    expected_return_w = objective_weights[0] * \
        utils.portfolio_expected_return(companies, portfolio_weights)
    risk_w = -1 * objective_weights[1] * utils.portfolio_risk(
        companies, portfolio_weights)
    values = [expected_return_w, risk_w]
    if len(objective_weights) == 3:
        included_companies_w = objective_weights[2] \
              * utils.portfolio_num_included_companies(portfolio_weights) \
              / len(companies)
        values.append(included_companies_w)
    return max(values)


def evaluate_portfolio_weighted_sum(
        companies: list[Company],
        portfolio_weights: Iterable[float],
        objective_weights: tuple[float]) -> float:
    """Implicit objective order: expected_return, risk, number of included
    companies"""
    expected_return_w = objective_weights[0] * \
        utils.portfolio_expected_return(companies, portfolio_weights)
    risk_w = -1 * objective_weights[1] * utils.portfolio_risk(
        companies, portfolio_weights)
    values = [expected_return_w, risk_w]
    if len(objective_weights) == 3:
        included_companies_w = objective_weights[2] \
              * utils.portfolio_num_included_companies(portfolio_weights) \
              / len(companies)
        values.append(included_companies_w)
    return sum(values)


def closest_goals(goal_vectors: Iterable[tuple[float]],
                  neighborhood_size: int
                  ) -> dict[tuple[float], tuple[tuple[float]]]:
    """Returns a dictionary (goal): (k closest goals, including itself)"""
    def distance(t1: tuple, t2: tuple) -> float:
        return np.sqrt(sum([(a-b)**2 for a, b in zip(t1, t2)]))
    closest_dict = dict()
    for goal1 in goal_vectors:
        objective_dist_pairs = []
        # the objective itself should be included in the neighborhood!
        for goal2 in goal_vectors:
            dist = distance(goal1, goal2)
            objective_dist_pairs.append((goal2, dist))
        objective_dist_pairs.sort(key=lambda x: x[1])
        closest_dict[goal1] = tuple(
            goal[0] for goal in objective_dist_pairs[:neighborhood_size])
    return closest_dict


def MOEAD_parent_selection(
        goal: tuple[float, float], portfolio_assignments: dict,
        goal_neighborhoods: dict) -> tuple[np.ndarray, np.ndarray]:
    neighbors = goal_neighborhoods[goal]
    indices = np.random.choice(range(len(neighbors)), 2, replace=False)
    selected_neighbors = (neighbors[indices[0]], neighbors[indices[1]])
    return (portfolio_assignments[selected_neighbors[0]],
            portfolio_assignments[selected_neighbors[1]])


def MOEAD_offspring(
        companies: list[Company],
        goal: tuple[float], portfolio_assignments: dict,
        goal_neighborhoods: dict, distribution_index: int) -> Iterable[float]:
    parent1, parent2 = MOEAD_parent_selection(goal, portfolio_assignments,
                                              goal_neighborhoods)
    # the crossover produces two offspring. From them, the one better
    # with respect to the goal is selected
    offspring1, offspring2 = evolutionary_operators.SBX_portfolios(
        parent1, parent2, distribution_index)
    fitness1 = evaluate_portfolio_weighted_sum(companies, offspring1, goal)
    fitness2 = evaluate_portfolio_weighted_sum(companies, offspring2, goal)
    if fitness1 > fitness2:
        return offspring1
    else:
        return offspring2


def sample_goal_weights(
        num_scalarizing_functions: int,
        num_objectives: int) -> list[tuple[float]]:
    sampled_weights = get_reference_directions(
        "energy", n_points=num_scalarizing_functions, n_dim=num_objectives)
    sampled_weights = [tuple(weights) for weights in sampled_weights]
    return sampled_weights


def assign_initial_pop_to_goals(
        companies: list[Company],
        population: np.ndarray[np.float32], fitness_function: Callable,
        goal_weights: list[tuple[float]]) -> tuple[dict, dict]:
    """Returns a pair of dicts:
    (goal): assigned portfolio
    (goal): the evaluation of the assigned portfolio w.r.t this goal
    """
    # no shuffling, since the generated population is random anyway
    portfolio_assignments = dict(zip(goal_weights, population))
    fitness_assignments = dict()
    for obj_weights, portfolio in portfolio_assignments.items():
        fitness = fitness_function(companies, portfolio, obj_weights)
        fitness_assignments[obj_weights] = fitness
    return portfolio_assignments, fitness_assignments


def MOEAD_main_loop(
        companies: list[Company], fitness_function_name: str,
        population_size: int = 100,
        n_objectives: int = 2, neighborhood_size: int = 3,
        generations: int = 500,
        crossover_distr_idx: int = 5,
        mutation_probability: float = 0.1
        ) -> tuple[np.ndarray[np.float32], int]:
    """Returns the final population and the number of the final generation"""
    # Initialization
    population = evolutionary_operators.random_portfolio_population(
        len(companies), population_size)
    evolutionary_operators.export_population(population, EXPORT_PATH, PARAMETERS, 0, "a+")
    # plot the initial population
    plot_population(companies, population, generation_rel=0, show=False, force_color="gray", alpha=0.5)
    sampled_weights = sample_goal_weights(population_size, n_objectives)
    # using a string to allow for consistent exporting
    match fitness_function_name:
        case "chebyshev":
            fitness_function = evaluate_portfolio_chebyshev
        case "weighted_sum":
            fitness_function = evaluate_portfolio_weighted_sum
    portfolio_assignments, fitness_assignments = assign_initial_pop_to_goals(
        companies, population, fitness_function, sampled_weights)
    goal_neighborhoods = closest_goals(sampled_weights, neighborhood_size)

    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring(
                companies, goal, portfolio_assignments, goal_neighborhoods,
                crossover_distr_idx)
            evolutionary_operators.mutate_portfolio(
                offspring, mutation_probability)
            for neighboring_goal in goal_neighborhoods[goal]:
                fitness = fitness_function(
                    companies, offspring, neighboring_goal)
                if fitness > fitness_assignments[neighboring_goal]:
                    portfolio_assignments[neighboring_goal] = offspring
                    fitness_assignments[neighboring_goal] = fitness
                    no_improvement_count = 0
                    improvement_this_iter = True
                    break  # unsure if this break should be here
        if not improvement_this_iter:
            no_improvement_count += 1
        if no_improvement_count == iter_without_improvement_cap:
            print(f"NO IMPROVEMENT IN GENERATION: {generation+1}")
            break
        if len(set([utils.portfolio_expected_return(companies, p) for p in portfolio_assignments.values()])) < population_size:
            print(f"REDUCED POP SIZE IN GENERATION: {generation+1}")
            break
        if generation % 10 == 0:
            np_pop = np.array(list(portfolio_assignments.values()))
            generation_rel = generation/generations
            evolutionary_operators.export_population(population, EXPORT_PATH, PARAMETERS, generation+1, "a+", True)
            plot_population(companies, np_pop, generation_rel, show=False, alpha=generation_rel)
    return np.array(list(portfolio_assignments.values())), generation


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


PARAMETERS = {
    "fitness_function_name": "chebyshev",
    "n_objectives": 2,
    "neighborhood_size": 3,
    "generations": 500,
    "population_size": 100,
    "crossover_distr_idx": 1,
    "mutation_probability": 0.1,
}


def path_from_params(parameters: dict) -> str:
    nhood = parameters["neighborhood_size"]
    xover = parameters["crossover_distr_idx"]
    mut = parameters["mutation_probability"]
    return f"./populations/nhood_{nhood}_xover_{xover}_mut_{mut}.txt"


EXPORT_PATH = path_from_params(PARAMETERS)


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    for company in companies:
        company.expected_return, _ = return_estimation.predict_expected_return_linear_regression(company, 200)
    pop, gen_num = MOEAD_main_loop(companies, **PARAMETERS)
    plot_population(companies, pop, 1, force_color="red")
    evolutionary_operators.export_population(pop, EXPORT_PATH, PARAMETERS, gen_num, "a+", True)

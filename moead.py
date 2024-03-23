import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from datetime import datetime

from company import Company
import data_loading
import problem_construction
import return_estimation
import evolutionary_operators
from evolutionary_visualizations import plot_population
import utils
from typing import Iterable, Callable


def evaluate_portfolio_chebyshev(
        companies: list[Company],
        portfolio_weights: Iterable[float],
        objective_weights: tuple[float],
        ret_norm_const: float,
        risk_norm_const: float) -> float:
    """Implicit objective order: expected_return, risk, number of included
    companies
    What is being maximized: NEGATIVE DISTANCE TO THE OPTIMUM"""
    expected_return_w = objective_weights[0] * \
        utils.portfolio_expected_return(companies, portfolio_weights) \
        / ret_norm_const
    # UGLY simplification: taking the return spread as the optimum value of return
    expected_return_w = -1 * (ret_norm_const - expected_return_w)
    risk_w = objective_weights[1] * utils.portfolio_risk(
        companies, portfolio_weights) / \
        risk_norm_const
    risk_w = -1 * risk_w  # optimum risk is 0
    values = [expected_return_w, risk_w]
    if len(objective_weights) == 3:
        included_companies_w = objective_weights[2] \
              * utils.portfolio_num_included_companies(portfolio_weights) \
              / len(companies)
        included_companies_w = -1 * (1-included_companies_w)
        values.append(included_companies_w)
    # must return the objective the solution is the worst at
    return min(values)


def evaluate_portfolio_weighted_sum(
        companies: list[Company],
        portfolio_weights: Iterable[float],
        objective_weights: tuple[float],
        ret_norm_const: float,
        risk_norm_const: float) -> float:
    """Implicit objective order: expected_return, risk, number of included
    companies"""
    expected_return_w = objective_weights[0] * \
        utils.portfolio_expected_return(companies, portfolio_weights) / \
        ret_norm_const
    risk_w = -1 * objective_weights[1] * utils.portfolio_risk(
        companies, portfolio_weights) / \
        risk_norm_const
    values = [expected_return_w, risk_w]
    if len(objective_weights) == 3:
        included_companies_w = objective_weights[2] \
              * utils.portfolio_num_included_companies(portfolio_weights) \
              / len(companies)
        values.append(included_companies_w)
    return sum(values)


def closest_goals(
        goal_vectors: Iterable[tuple[float]],
        neighborhood_size: int) -> dict[tuple[float], tuple[tuple[float]]]:
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
        goal_neighborhoods: dict, crossover_mode: float,
        distribution_index: int,
        fitness_function: Callable,
        ret_norm_const: float, risk_norm_const: float) -> Iterable[float]:
    parent1, parent2 = MOEAD_parent_selection(
        goal, portfolio_assignments, goal_neighborhoods)
    # the crossover produces two offspring. From them, the one better
    # with respect to the goal is selected
    offspring1, offspring2 = evolutionary_operators.SBX_portfolios(
        parent1, parent2, crossover_mode, distribution_index)
    fitness1 = fitness_function(companies, offspring1, goal, ret_norm_const, risk_norm_const)
    fitness2 = fitness_function(companies, offspring2, goal, ret_norm_const, risk_norm_const)
    if fitness1 > fitness2:
        return offspring1
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
        ret_norm_const: float, risk_norm_const: float,
        goal_weights: list[tuple[float]]) -> tuple[dict, dict]:
    """Returns a pair of dicts:
    (goal): assigned portfolio
    (goal): the evaluation of the assigned portfolio w.r.t this goal
    """
    # no shuffling, since the generated population is random anyway
    portfolio_assignments = dict(zip(goal_weights, population))
    fitness_assignments = dict()
    for obj_weights, portfolio in portfolio_assignments.items():
        fitness = fitness_function(
            companies, portfolio, obj_weights, ret_norm_const, risk_norm_const)
        fitness_assignments[obj_weights] = fitness
    return portfolio_assignments, fitness_assignments


def MOEAD_main_loop(
        companies: list[Company],
        export_path: str,
        export_params_dict: dict,
        ret_norm_const: float,
        risk_norm_const: float,
        fitness_function_name: str,
        population_size: int = 100,
        n_objectives: int = 2, neighborhood_size: int = 3,
        generations: int = 500,
        crossover_distr_idx: int = 5,
        crossover_mode: float = 0.9,
        mutation_probability: float = 0.1
        ) -> tuple[np.ndarray[np.float32], int]:
    """Returns the final population and the number of the final generation"""
    # Initialization
    population = evolutionary_operators.random_portfolio_population(
        len(companies), population_size)
    evolutionary_operators.export_population(population, export_path, export_params_dict, 0, "a+")
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
        companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    goal_neighborhoods = closest_goals(sampled_weights, neighborhood_size)

    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring(
                companies, goal, portfolio_assignments, goal_neighborhoods, crossover_mode,
                crossover_distr_idx, fitness_function, ret_norm_const, risk_norm_const)
            evolutionary_operators.mutate_portfolio(
                offspring, mutation_probability)
            for neighboring_goal in goal_neighborhoods[goal]:
                fitness = fitness_function(
                    companies, offspring, neighboring_goal, ret_norm_const, risk_norm_const)
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
        np_pop = np.array(list(portfolio_assignments.values()))
        num_unique = evolutionary_operators.num_unique_individuals_in_pop(np_pop)
        if num_unique < population_size:
            print(f"REDUCED POP SIZE IN GENERATION: {generation+1}")
            print(f"Want: {population_size}, have: {num_unique}")
            break
        if generation % 10 == 0:
            generation_rel = generation/generations
            evolutionary_operators.export_population(np_pop, export_path, export_params_dict, generation+1, "a+", True)
            plot_population(companies, np_pop, generation_rel, show=False, alpha=generation_rel)
    return np.array(list(portfolio_assignments.values())), generation


def minimal_MOEAD_loop(
        companies: list[Company],
        ret_norm_const: float,
        risk_norm_const: float,
        fitness_function_name: str,
        population_size: int = 100,
        n_objectives: int = 2, neighborhood_size: int = 3,
        generations: int = 500,
        crossover_mode: float = 0.9,
        crossover_distr_idx: int = 5,
        mutation_probability: float = 0.1
        ) -> tuple[np.ndarray[np.float32], list[int]]:
    # Initialization
    population = evolutionary_operators.random_portfolio_population(
        len(companies), population_size)
    gen_list = [0] * population_size
    populations = population.copy()
    sampled_weights = sample_goal_weights(population_size, n_objectives)
    # using a string to allow for consistent exporting
    match fitness_function_name:
        case "chebyshev":
            fitness_function = evaluate_portfolio_chebyshev
        case "weighted_sum":
            fitness_function = evaluate_portfolio_weighted_sum
    portfolio_assignments, fitness_assignments = assign_initial_pop_to_goals(
        companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    goal_neighborhoods = closest_goals(sampled_weights, neighborhood_size)

    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring(
                companies, goal, portfolio_assignments, goal_neighborhoods, crossover_mode,
                crossover_distr_idx, fitness_function, ret_norm_const, risk_norm_const)
            evolutionary_operators.mutate_portfolio(
                offspring, mutation_probability)
            for neighboring_goal in goal_neighborhoods[goal]:
                fitness = fitness_function(
                    companies, offspring, neighboring_goal, ret_norm_const, risk_norm_const)
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
        np_pop = np.array(list(portfolio_assignments.values()))
        num_unique = evolutionary_operators.num_unique_individuals_in_pop(np_pop)
        if num_unique < population_size:
            print(f"REDUCED POP SIZE IN GENERATION: {generation+1}")
            print(f"Want: {population_size}, have: {num_unique}")
            break
        if generation % 10 == 0:
            populations = np.vstack((populations, np_pop))
            gen_list += [generation] * population_size
    if gen_list[-1] == generation:
        point_array = np.array([(utils.portfolio_expected_return(companies, p), utils.portfolio_risk(companies, p)) for p in populations])
        return point_array, generations
    np_pop = np.array(list(portfolio_assignments.values()))
    populations = np.vstack((populations, np_pop))
    point_array = np.array([(utils.portfolio_expected_return(companies, p), utils.portfolio_risk(companies, p)) for p in populations])
    gen_list += [generation] * population_size
    return point_array, gen_list


def MOEAD_experiment(companies: list[Company], num_runs: int, parameters: dict) -> None:
    ret_norm_const, risk_norm_const = problem_construction.exp_ret_risk_spreads(companies)
    points = np.array([[0,0]],  dtype=np.float32)
    gens = []
    for i in range(num_runs):
        print(f"Starting run {i+1}/{num_runs}... {datetime.now().strftime('%H:%M:%S')}")
        p, g = minimal_MOEAD_loop(companies, ret_norm_const, risk_norm_const, **parameters)
        points = np.vstack((points, p))
        gens += g
    exp_path = experiment_path_from_params(parameters)
    # points[1:] to skip the initial zeros
    evolutionary_operators.export_population_points(points[1:], exp_path, parameters, gens)


PARAMETERS = {
    "fitness_function_name": "weighted_sum",  # "chebyshev" or "weighted_sum"
    "n_objectives": 3,
    "neighborhood_size": 3,
    "generations": 500,
    "population_size": 100,
    "crossover_distr_idx": 1,
    "crossover_mode": 0.9,
    "mutation_probability": 0.1,
}


def path_from_params(parameters: dict) -> str:
    fname = parameters['fitness_function_name']
    nhood = parameters["neighborhood_size"]
    xover = parameters["crossover_distr_idx"]
    mut = parameters["mutation_probability"]
    return f"./populations/{fname}_nhood_{nhood}_xover_{xover}_mut_{mut}.txt"


def experiment_path_from_params(parameters: dict) -> str:
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"./populations/EXPERIMENT_{time_str}.txt"


EXPORT_PATH = path_from_params(PARAMETERS)
RET_NORM_CONST = 1.01
RISK_NORM_CONST = 0.25

if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    for company in companies:
        company.expected_return, _ = return_estimation.predict_expected_return_linear_regression(company, 200)
    # pop, gen_num = MOEAD_main_loop(companies, EXPORT_PATH, PARAMETERS, RET_NORM_CONST, RISK_NORM_CONST, **PARAMETERS)
    # plot_population(companies, pop, 1, force_color="red")
    # evolutionary_operators.export_population(pop, EXPORT_PATH, PARAMETERS, gen_num, "a+", True)
    MOEAD_experiment(companies, 10, PARAMETERS)

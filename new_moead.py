import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from datetime import datetime

from company import Company
import data_loading
import problem_construction
import return_estimation
import evolutionary_operators
from evolutionary_visualizations import plot_population
import moead
import utils
from typing import Iterable, Callable


def MOEAD_hold_soft_tournament(
        goal: tuple[float, float], goal_list: list[tuple[float, float, float]],
        portfolio_assignments: dict, all_fitness_dict: dict,
        tournament_size: int, neighborhood_size: int) -> np.ndarray:
    goal_idx = goal_list.index(goal)
    # randomly chose which solutions take part in the tournament
    participants_indices = np.random.choice(list(range(len(goal_list))), tournament_size, replace=False)
    participants = [goal_list[idx] for idx in participants_indices]
    # find fitness values with respect to the selected goal
    fitness_values = [all_fitness_dict[p][goal_idx] for p in participants]
    order = np.argsort(fitness_values)
    indices_of_best = order[-neighborhood_size:]  # used to be -neighborhood_size; -(tournament_size//2)
    # pick the best solution at this goal
    # goal_of_best_sol = goal_list[np.argmax(fitness_values)]
    # randomly pick among some best solutions
    goal_of_best_sol = goal_list[np.random.choice(indices_of_best)]
    return portfolio_assignments[goal_of_best_sol]


def MOEAD_tournament_parent_selection(
        goal: tuple[float, float], portfolio_assignments: dict,
        all_fitness_dict: dict, tournament_size: int, neighborhood_size: int
        ) -> tuple[np.ndarray, np.ndarray]:
    goal_list = list(portfolio_assignments.keys())
    # introduce a new dict - one that maps tuple -> list/np.ndarray
    # tuple is used to identify the solution in the current population (the solution can be found by looking up the same key in portfolio_assignments)
    # since there is an order to items in python dictionaries, the ith position of the list will correspond to the fitness of the solution on the ith goal
    parent1 = MOEAD_hold_soft_tournament(goal, goal_list, portfolio_assignments, all_fitness_dict, tournament_size, neighborhood_size)
    parent2 = MOEAD_hold_soft_tournament(goal, goal_list, portfolio_assignments, all_fitness_dict, tournament_size, neighborhood_size)
    return parent1, parent2


def create_all_goals_fitness_dict(
        portfolio_assignments: dict, fitness_assignments: dict,
        fitness_function: Callable, companies:  list[Company],
        ret_norm_const: float, risk_norm_const: float
        ) -> dict[tuple, np.ndarray[np.float32]]:
    # fitness_dict is used to slightly speed up computation - solutions already have fitness calculated for the goal to which they are assigned
    all_fitness_dict = dict(zip(portfolio_assignments.keys(), [np.zeros(len(portfolio_assignments.keys())) for k in portfolio_assignments.keys()]))
    for goal in portfolio_assignments.keys():
        for i, other_goal in enumerate(portfolio_assignments.keys()):
            if goal == other_goal:
                all_fitness_dict[goal][i] = fitness_assignments[goal]
                continue
            fitness = fitness_function(companies, portfolio_assignments[goal], other_goal, ret_norm_const, risk_norm_const)
            all_fitness_dict[goal][i] = fitness
    return all_fitness_dict


def update_dicts_after_new_sol(
        portfolio_weights: Iterable[float],
        assigned_to: tuple, assigned_fitness: float,
        portfolio_assignments: dict,
        fitness_assignments: dict, all_fitness_dict: dict) -> None:
    portfolio_assignments[assigned_to] = portfolio_weights
    fitness_assignments[assigned_to] = assigned_fitness
    for i, goal in enumerate(portfolio_assignments.keys()):
        if goal == assigned_to:
            all_fitness_dict[goal][i] = assigned_fitness
            continue
        all_fitness_dict[goal][i] = assigned_fitness
    return None


def MOEAD_offspring_tournament_selection(
        companies: list[Company],
        goal: tuple[float], portfolio_assignments: dict,
        all_fitness_dict: dict, tournament_size: int,
        crossover_mode: float,
        distribution_index: int,
        fitness_function: Callable,
        ret_norm_const: float, risk_norm_const: float,
        neighborhood_size: int) -> Iterable[float]:
    parent1, parent2 = MOEAD_tournament_parent_selection(
        goal, portfolio_assignments, all_fitness_dict, tournament_size, neighborhood_size)
    # the crossover produces two offspring. From them, the one better
    # with respect to the goal is selected
    offspring1, offspring2 = evolutionary_operators.SBX_portfolios(
        parent1, parent2, crossover_mode, distribution_index)
    # offspring1, offspring2 = evolutionary_operators.SBX_more_random(
    #     parent1, parent2, crossover_mode, distribution_index)
    fitness1 = fitness_function(companies, offspring1, goal, ret_norm_const, risk_norm_const)
    fitness2 = fitness_function(companies, offspring2, goal, ret_norm_const, risk_norm_const)
    if fitness1 > fitness2:
        return offspring1
    return offspring2


def MOEAD_offspring_more_random(
        companies: list[Company],
        goal: tuple[float], portfolio_assignments: dict,
        goal_neighborhoods: dict, crossover_mode: float,
        distribution_index: int,
        fitness_function: Callable,
        ret_norm_const: float, risk_norm_const: float) -> Iterable[float]:
    parent1, parent2 = moead.MOEAD_parent_selection(
        goal, portfolio_assignments, goal_neighborhoods)
    # the crossover produces two offspring. From them, the one better
    # with respect to the goal is selected
    offspring1, offspring2 = evolutionary_operators.SBX_more_random(parent1, parent2, crossover_mode, distribution_index)
    fitness1 = fitness_function(companies, offspring1, goal, ret_norm_const, risk_norm_const)
    fitness2 = fitness_function(companies, offspring2, goal, ret_norm_const, risk_norm_const)
    if fitness1 > fitness2:
        return offspring1
    return offspring2


def MOEAD_tournament_selection_main_loop(
        companies: list[Company],
        export_path: str,
        export_params_dict: dict,
        tournament_size: int,
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
    sampled_weights = moead.sample_goal_weights(population_size, n_objectives)
    # using a string to allow for consistent exporting
    match fitness_function_name:
        case "chebyshev":
            fitness_function = moead.evaluate_portfolio_chebyshev
        case "weighted_sum":
            fitness_function = moead.evaluate_portfolio_weighted_sum
    portfolio_assignments, fitness_assignments = moead.assign_initial_pop_to_goals(
        companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    goal_neighborhoods = moead.closest_goals(sampled_weights, neighborhood_size)
    all_fitness_dict = create_all_goals_fitness_dict(portfolio_assignments, fitness_assignments, fitness_function, companies, ret_norm_const, risk_norm_const)

    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring_tournament_selection(
                companies, goal, portfolio_assignments, all_fitness_dict, tournament_size, crossover_mode,
                crossover_distr_idx, fitness_function, ret_norm_const, risk_norm_const, neighborhood_size)
            evolutionary_operators.mutate_portfolio(
                offspring, mutation_probability)
            for neighboring_goal in goal_neighborhoods[goal]:
                fitness = fitness_function(
                    companies, offspring, neighboring_goal, ret_norm_const, risk_norm_const)
                if fitness > fitness_assignments[neighboring_goal]:
                    update_dicts_after_new_sol(offspring, goal, fitness, portfolio_assignments, fitness_assignments, all_fitness_dict)
                    no_improvement_count = 0
                    improvement_this_iter = True
                    break  # unsure if this break should be here
        if not improvement_this_iter:
            no_improvement_count += 1
        if no_improvement_count == iter_without_improvement_cap:
            print(f"NO IMPROVEMENT IN GENERATION: {generation+1}")
            break
        np_pop = np.array(list(portfolio_assignments.values()))
        if generation % 10 == 0:
            generation_rel = generation/generations
            evolutionary_operators.export_population(np_pop, export_path, export_params_dict, generation+1, "a+", True)
            plot_population(companies, np_pop, generation_rel, show=False, alpha=generation_rel)
    return np.array(list(portfolio_assignments.values())), generation


def MOEAD_more_exploratory_recombination_main_loop(
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
    sampled_weights = moead.sample_goal_weights(population_size, n_objectives)
    # using a string to allow for consistent exporting
    match fitness_function_name:
        case "chebyshev":
            fitness_function = moead.evaluate_portfolio_chebyshev
        case "weighted_sum":
            fitness_function = moead.evaluate_portfolio_weighted_sum
    portfolio_assignments, fitness_assignments = moead.assign_initial_pop_to_goals(
        companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)

    goal_neighborhoods = moead.closest_goals(sampled_weights, neighborhood_size)
    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring_more_random(companies, goal, portfolio_assignments, goal_neighborhoods, crossover_mode, crossover_distr_idx, fitness_function,  ret_norm_const, risk_norm_const)
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
        # num_unique = evolutionary_operators.num_unique_individuals_in_pop(np_pop)
        # if num_unique < population_size:
        #     print(f"REDUCED POP SIZE IN GENERATION: {generation+1}")
        #     print(f"Want: {population_size}, have: {num_unique}")
        #     break
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
    # population = evolutionary_operators.random_portfolio_population(
        # len(companies), population_size)
    gen_list = [0] * population_size
    # populations = population.copy()
    sampled_weights = moead.sample_goal_weights(population_size, n_objectives)
    # using a string to allow for consistent exporting
    match fitness_function_name:
        case "chebyshev":
            fitness_function = moead.evaluate_portfolio_chebyshev
        case "weighted_sum":
            fitness_function = moead.evaluate_portfolio_weighted_sum

    # portfolio_assignments, fitness_assignments = assign_initial_pop_to_goals(
    #     companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    portfolio_assignments, fitness_assignments = moead.assign_initial_pop_to_goals(
        companies, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    goal_neighborhoods = moead.closest_goals(sampled_weights, neighborhood_size)
    populations = np.array(list(portfolio_assignments.values()))
    # populations = np.array([portfolio_assignments[k] for k in portfolio_assignments.keys()])
    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = moead.MOEAD_offspring(
                companies, goal, portfolio_assignments, goal_neighborhoods, crossover_mode,
                crossover_distr_idx, fitness_function, ret_norm_const, risk_norm_const)
            # offspring = MOEAD_offspring_more_random(
            #     companies, goal, portfolio_assignments, goal_neighborhoods, crossover_mode,
            #     crossover_distr_idx, fitness_function, ret_norm_const, risk_norm_const)
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
            print(len(portfolio_assignments.keys()))
            print(f"REDUCED POP SIZE IN GENERATION: {generation+1}")
            print(f"Want: {population_size}, have: {num_unique}\t(shape: {np_pop.shape})")
            return
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


def minimal_MOEAD_tournament_selection_loop(
        companies: list[Company],
        tournament_size: int,
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
    sampled_weights = moead.sample_goal_weights(population_size, n_objectives)
    # using a string to allow for consistent exporting
    match fitness_function_name:
        case "chebyshev":
            fitness_function = moead.evaluate_portfolio_chebyshev
        case "weighted_sum":
            fitness_function = moead.evaluate_portfolio_weighted_sum

    # portfolio_assignments, fitness_assignments = assign_initial_pop_to_goals(
    #     companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    portfolio_assignments, fitness_assignments = moead.assign_initial_pop_to_goals(
        companies, population, fitness_function, ret_norm_const, risk_norm_const, sampled_weights)
    goal_neighborhoods = moead.closest_goals(sampled_weights, neighborhood_size)
    all_fitness_dict = create_all_goals_fitness_dict(portfolio_assignments, fitness_assignments, fitness_function, companies, ret_norm_const, risk_norm_const)
    populations = np.array(list(portfolio_assignments.values()))
    # populations = np.array([portfolio_assignments[k] for k in portfolio_assignments.keys()])
    # Loop helper variables
    iter_without_improvement_cap: int = 3
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generations):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring_tournament_selection(
                companies, goal, portfolio_assignments, all_fitness_dict, tournament_size, crossover_mode,
                crossover_distr_idx, fitness_function, ret_norm_const, risk_norm_const, neighborhood_size)
            evolutionary_operators.mutate_portfolio(
                offspring, mutation_probability)
            for neighboring_goal in goal_neighborhoods[goal]:
                fitness = fitness_function(
                    companies, offspring, neighboring_goal, ret_norm_const, risk_norm_const)
                if fitness > fitness_assignments[neighboring_goal]:
                    update_dicts_after_new_sol(offspring, neighboring_goal, fitness, portfolio_assignments, fitness_assignments, all_fitness_dict)
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
            print(len(portfolio_assignments.keys()))
            print(f"REDUCED POP SIZE IN GENERATION: {generation+1}")
            print(f"Want: {population_size}, have: {num_unique}\t(shape: {np_pop.shape})")
            return
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
    exp_path = experiment_path_from_params(parameters)
    completed_runs = 0
    attempted_runs = 0
    # for i in range(num_runs):
    while completed_runs < num_runs:
        print(f"Attempting run {attempted_runs+1}. Completed: {completed_runs}/{num_runs}... {datetime.now().strftime('%H:%M:%S')}")
        try:
            attempted_runs+=1
            p, g = minimal_MOEAD_loop(companies, ret_norm_const, risk_norm_const, **parameters)
        except TypeError:
            continue
        completed_runs+=1
        evolutionary_operators.export_population_points(p, exp_path, parameters, g)
        points = np.vstack((points, p))
        gens += g
    # points[1:] to skip the initial zeros
    # evolutionary_operators.export_population_points(points[1:], exp_path, parameters, gens)


def MOEAD_tournament_experiment(companies: list[Company], num_runs: int, parameters: dict) -> None:
    ret_norm_const, risk_norm_const = problem_construction.exp_ret_risk_spreads(companies)
    points = np.array([[0,0]],  dtype=np.float32)
    gens = []
    exp_path = experiment_path_from_params(parameters)
    tournament_size = parameters["population_size"] - parameters["neighborhood_size"] + 1
    completed_runs = 0
    attempted_runs = 0
    # for i in range(num_runs):
    while completed_runs < num_runs:
        print(f"Attempting run {attempted_runs+1}. Completed: {completed_runs}/{num_runs}... {datetime.now().strftime('%H:%M:%S')}")
        try:
            attempted_runs += 1
            p, g = minimal_MOEAD_tournament_selection_loop(companies, tournament_size, ret_norm_const, risk_norm_const, **parameters)
        except TypeError:
            continue
        completed_runs+=1
        evolutionary_operators.export_population_points(p, exp_path, parameters, g)
        points = np.vstack((points, p))
        gens += g


PARAMETERS = {
    "fitness_function_name": "weighted_sum",  # "chebyshev" or "weighted_sum"
    "n_objectives": 2,
    "neighborhood_size": 3,
    "generations": 50,
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
    gs = [500, 1000]
    # ps = [50, 100, 200]
    ps = [200]
    for i, g in enumerate(gs):
        for j, p in enumerate(ps):
            PARAMETERS["generations"] = g
            PARAMETERS["population_size"] = p
            print(f"[{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}] Starting experiment pop_size: {p} generations: {g}")
            MOEAD_tournament_experiment(companies, 10, PARAMETERS)
    # assignment_dict, fitness_dict = assign_initial_pop_to_goals(companies, evolutionary_operators.random_portfolio_population(20, 100), evaluate_portfolio_chebyshev, RET_NORM_CONST, RISK_NORM_CONST, sample_goal_weights(100, 2))
    # f_dict = create_all_goals_fitness_dict(assignment_dict, fitness_dict, evaluate_portfolio_weighted_sum, companies, RET_NORM_CONST, RISK_NORM_CONST)
    # print(f_dict)
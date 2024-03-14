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
        companies: list[Company], fitness_function: Callable,
        num_scalarizing_functions: int = 100,
        num_objectives: int = 2, neighborhood_size: int = 3,
        generation_cap: int = 500, iter_without_improvement_cap: int = 3,
        crossover_distr_index: int = 5,
        mutation_probability: float = 0.1
        ) -> np.ndarray[np.float32]:
    """Returns the final population"""
    # Initialization
    population = evolutionary_operators.random_portfolio_population(
        len(companies), num_scalarizing_functions)
    sampled_weights = sample_goal_weights(num_scalarizing_functions, num_objectives)
    portfolio_assignments, fitness_assignments = assign_initial_pop_to_goals(
        population, fitness_function, sampled_weights)
    goal_neighborhoods = closest_goals(sampled_weights, neighborhood_size)

    # Loop helper variables
    no_improvement_count = 0
    improvement_this_iter = False
    for generation in range(generation_cap):
        improvement_this_iter = False
        for goal in sampled_weights:
            offspring = MOEAD_offspring(
                goal, portfolio_assignments, goal_neighborhoods,
                crossover_distr_index)
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
        if generation % 100 == 0:
            np_pop = np.array(list(portfolio_assignments.values()))
            generation_rel = generation/generation_cap
            plot_population(companies, np_pop, generation_rel, show=False, alpha=generation_rel)
    return np.array(list(portfolio_assignments.values()))


def plot_population(
        companies: list[Company],
        population: np.ndarray[np.float32],
        generation_rel: float,
        history_len: int | None = None,
        title: str = "",
        export_pdf: bool = False,
        pdf_title: str = "pop1.pdf",
        show: bool = True, alpha: float = 1.0) -> None:
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

    plt.plot(returns, risks, "o", alpha=alpha, c=cm.viridis(generation_rel))
    plt.xlabel("Expected return [100%]", **label_font)
    plt.ylabel("Risk [$ \$^2 $] ", **label_font)
    plt.grid()
    plt.title(title, **label_font)
    if export_pdf:
        plt.savefig(pdf_title, format="pdf")
    if show:
        plt.show()


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    for company in companies:
        company.expected_return, _ = return_estimation.predict_expected_return_linear_regression(company, 200)
    pop = MOEAD_main_loop(companies, evaluate_portfolio_chebyshev, 100, 2, generation_cap=500)
    plot_population(companies, pop, 1)

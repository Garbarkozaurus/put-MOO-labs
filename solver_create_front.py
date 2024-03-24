import matplotlib.pyplot as plt

from company import Company
import data_loading
import return_estimation
import problem_construction
import utils


def uniformly_search_weight_space(
        companies: list[Company],
        num_weights: int,
        prediction_time: int,
        fitting_timeline_start: int = 0,
        history_len: int | None = None,
        calculate_expected_return: bool = True) -> list[tuple[dict, dict]]:
    """Portfolio optimization using the weighted-sum method
    Uses a linear regression model to predict the expected return
    Returns a list of two-element tuples:
    - the first element is a dictionary {'expected_return_weight': a, 'risk_weight': b}
    - the second is the dictionary returned by the appropriate cvxopt.solver"""
    if calculate_expected_return:
        for company in companies:
            company.expected_return, _ = return_estimation.predict_expected_return_linear_regression(
                company, prediction_time, fitting_timeline_start)
    max_ret_solution = problem_construction.maximize_return_solver(companies)
    min_risk_solution = problem_construction.minimize_risk_solver(companies, history_len)
    ret_spread, risk_spread = problem_construction.exp_ret_risk_spreads(companies)
    solutions = [({"expected_return_weight": 0.0, "risk_weight": 1.0}, min_risk_solution)]
    for i in range(num_weights-2):
        expected_return_weight = (i+1)/(num_weights-1)
        risk_weight = 1.0 - expected_return_weight
        solution = problem_construction.weighted_sum_solver(
            companies, expected_return_weight, risk_weight, ret_spread,
            risk_spread, history_len)
        solutions.append(({"expected_return_weight": expected_return_weight, "risk_weight": risk_weight}, solution))
    solutions.append(({"expected_return_weight": 1.0, "risk_weight": 0.0}, max_ret_solution))
    return solutions


def uniformly_search_threshold_space(
        companies: list[Company],
        num_thresholds: int,
        prediction_time: int,
        fitting_timeline_start: int = 0,
        history_len: int | None = None,
        calculate_expected_return: bool = True) -> list[tuple[float, dict]]:
    """Portfolio optimization using the epsilon-constrained method
    Uses a linear regression model to predict the expected return
    Returns a list of two-element tuples:
    - the first element is the value of the minimum expected return threshold
    - the second is the dictionary returned by the appropriate cvxopt.solver"""
    if calculate_expected_return:
        for company in companies:
            company.expected_return, _ = return_estimation.predict_expected_return_linear_regression(
                company, prediction_time, fitting_timeline_start)
    max_ret_solution = problem_construction.maximize_return_solver(companies)
    max_ret = utils.portfolio_expected_return(companies, max_ret_solution['x'])
    min_risk_solution = problem_construction.minimize_risk_solver(companies, history_len)
    min_risk_ret = utils.portfolio_expected_return(companies, min_risk_solution['x'])
    solutions = [(0.0, min_risk_solution)]
    step_size = (max_ret - min_risk_ret) / num_thresholds
    minimum_return = min_risk_ret
    for _ in range(num_thresholds-1):
        # minimum_return = (i+1)*(max_ret) / num_thresholds
        solution = problem_construction.epsilon_constrained_solver(
            companies, minimum_return, history_len)
        solutions.append((minimum_return, solution))
        minimum_return += step_size
    solutions.append((max_ret, max_ret_solution))
    return solutions


def plot_front(companies: list[Company],
               solutions: list[tuple[dict, dict]],
               history_len: int | None = None,
               title: str = "",
               export_pdf: bool = False,
               pdf_title: str = "front.pdf") -> None:
    plot_points = []
    returns = []
    risks = []
    label_font = {'fontname': 'Times New Roman'}
    for _, sol in solutions:
        exp_ret = utils.portfolio_expected_return(companies, sol["x"])
        risk = utils.portfolio_risk(companies, sol["x"], history_len)
        plot_points.append((exp_ret, risk))
        returns.append(exp_ret)
        risks.append(risk)

    plt.plot(returns, risks, "ro")
    plt.xlabel("Expected return [100%]", **label_font)
    plt.ylabel("Risk [$ \$^2 $] ", **label_font)
    plt.grid()
    plt.title(title, **label_font)
    if export_pdf:
        plt.savefig(pdf_title, format="pdf")
    plt.show()


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    weighted_sum_solutions = uniformly_search_weight_space(companies, 100, 200)
    plot_front(companies, weighted_sum_solutions, title="Weighted sum")
    epsilon_solutions = uniformly_search_threshold_space(companies, 100, 200)
    plot_front(companies, epsilon_solutions, title="Epsilon constrained")

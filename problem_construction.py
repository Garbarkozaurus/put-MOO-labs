import numpy as np
import cvxopt

from company import Company
import data_loading
import utils


def nonnegativity_constraints_matrices(n: int) -> tuple[cvxopt.matrix]:
    """Returns a pair of matrices:
    - coefficients of non-negativity constraints
    (a diagonal n x n matrix with -1 as diagonal value to model >=)
    - bounds of non-negativity constraints (n repeats of 0.0)
    """
    # every row in the argument of cvxopt.matrix is a column
    # representing the coefficients for a variable in all constraints
    nonnegativity_constraints_coefs = cvxopt.matrix(np.diagflat([-1.0] * n))
    nonnegativity_constraints_bounds = cvxopt.matrix([0.0] * n)
    return nonnegativity_constraints_coefs, nonnegativity_constraints_bounds


def normalization_constraint_matrices(n: int) -> tuple[cvxopt.matrix]:
    """Returns a pair of matrices:
    - coefficients of the normalization constraint
    (a 1xn matrix of 1.0)
    - bounds of non-negativity constraints (a single 1.0)
    """
    normalization_constraint_coefs = cvxopt.matrix([[1.0] for _ in range(n)])
    normalization_constraint_bound = cvxopt.matrix([1.0])
    return normalization_constraint_coefs, normalization_constraint_bound


def maximize_return_solver(companies: list[Company]):
    """IMPORTANT: this function assumes that the values of expected return
    have already been calculated and stored in the expected_return
    field of appropriate companies

    The returned solver minimizes the negative return, but
    it is equivalent to maximizing the return (and it's more intuitive
    to think about it that way)"""
    # the solver just finds the company with the highest expected return
    # and assigns it a weight of 1, and 0 to every other
    # this makes perfect sense, and might make one think that this function
    # is redundant, but it serves as a neat example of defining a cvxopt solver
    expected_returns = [-1*company.expected_return for company in companies]
    exp_ret_matrix = cvxopt.matrix(expected_returns)

    num_companies = len(companies)
    # nonnegativity_constraints_bounds = cvxopt.matrix([0.0] * len(companies))
    nonnegativity_constraints_coefs, nonnegativity_constraints_bounds = \
        nonnegativity_constraints_matrices(num_companies)
    normalization_constraint_coefs, normalization_constraint_bound = \
        normalization_constraint_matrices(num_companies)
    return cvxopt.solvers.lp(exp_ret_matrix, nonnegativity_constraints_coefs,
                             nonnegativity_constraints_bounds,
                             normalization_constraint_coefs,
                             normalization_constraint_bound,
                             options={"show_progress": False})


def minimize_risk_solver(companies: list[Company]):
    covariance_matrix = utils.covariance_matrix_from_companies(companies)
    risk_matrix = cvxopt.matrix(covariance_matrix)
    # coefficients for the linear component of the optimized function
    c = cvxopt.matrix([0.0 for _ in companies])
    num_companies = len(companies)
    nonnegativity_constraints_coefs, nonnegativity_constraints_bounds = \
        nonnegativity_constraints_matrices(num_companies)
    normalization_constraint_coefs, normalization_constraint_bound = \
        normalization_constraint_matrices(num_companies)
    return cvxopt.solvers.qp(risk_matrix, c, nonnegativity_constraints_coefs,
                             nonnegativity_constraints_bounds,
                             normalization_constraint_coefs,
                             normalization_constraint_bound,
                             options={"show_progress": False})


def exp_ret_risk_spreads(
        companies: list[Company],
        history_len: int | None = None) -> tuple[float, float]:
    """Returns tuple: (expected_return_spread, risk_spread)"""
    solver_max_ret = maximize_return_solver(companies)
    max_ret_ret = -1*solver_max_ret["primal objective"]
    max_ret_weights = np.array(solver_max_ret["x"]).flatten()
    max_ret_risk = utils.portfolio_risk(companies, max_ret_weights, history_len)
    solver_min_risk = minimize_risk_solver(companies)
    min_risk_risk = solver_min_risk["primal objective"]
    min_risk_weights = np.array(solver_min_risk["x"]).flatten()
    min_risk_ret = utils.portfolio_expected_return(companies, min_risk_weights)
    ret_spread = max_ret_ret - min_risk_ret
    risk_spread = max_ret_risk - min_risk_risk
    return ret_spread, risk_spread


def weighted_sum_solver(
        companies: list[Company],
        ret_weight: float,
        risk_weight: float,
        ret_spread: float,
        risk_spread: float,
        history_len: int | None = None) -> cvxopt.solvers.qp:
    """IMPORTANT: this function assumes that the values of expected return
    have already been calculated and stored in the expected_return
    field of appropriate companies"""
    # Objective function
    # remember to normalize by dividing over the spread!
    expected_returns = [-1*company.expected_return for company in companies]
    exp_ret_matrix = cvxopt.matrix(expected_returns)
    exp_ret_matrix /= ret_spread
    exp_ret_matrix *= ret_weight
    covariance_matrix = utils.covariance_matrix_from_companies(companies, history_len)
    risk_matrix = cvxopt.matrix(covariance_matrix)
    risk_matrix /= risk_spread
    risk_matrix *= risk_weight

    # Constraints
    num_companies = len(companies)
    nonnegativity_constraints_coefs, nonnegativity_constraints_bounds = \
        nonnegativity_constraints_matrices(num_companies)
    normalization_constraint_coefs, normalization_constraint_bound = \
        normalization_constraint_matrices(num_companies)
    return cvxopt.solvers.qp(
        risk_matrix, exp_ret_matrix,
        nonnegativity_constraints_coefs, nonnegativity_constraints_bounds,
        normalization_constraint_coefs, normalization_constraint_bound,
        options={"show_progress": False})


def minimum_expected_return_constraint_matrices(
        companies: list[Company],
        min_ret: float) -> tuple[cvxopt.matrix]:
    """Returns a pair of matrices:
    - coefficients of the minimum expected return constraint
    (a 1xn matrix of expected return values for companies multiplied by -1
    to model >=)
    - bounds of non-negativity constraints (the value of min_ret)
    """
    constraint_coefs = cvxopt.matrix([[-1.0 * company.expected_return]
                                      for company in companies])
    min_ret_bound = cvxopt.matrix([min_ret])
    return constraint_coefs, min_ret_bound


def epsilon_constrained_solver(
        companies: list[Company],
        min_ret: float,
        history_len: int | None = None) -> cvxopt.solvers.qp:
    covariance_matrix = utils.covariance_matrix_from_companies(companies, history_len)
    risk_matrix = cvxopt.matrix(covariance_matrix)
    # coefficients for the linear component of the optimized function
    c = cvxopt.matrix([0.0 for _ in companies])
    num_companies = len(companies)
    nonnegativity_constraints_coefs, nonnegativity_constraints_bounds = \
        nonnegativity_constraints_matrices(num_companies)
    normalization_constraint_coefs, normalization_constraint_bound = \
        normalization_constraint_matrices(num_companies)
    min_ret_coefs, min_ret_bound = \
        minimum_expected_return_constraint_matrices(companies, min_ret)
    inequality_coefs = np.vstack((np.array(nonnegativity_constraints_coefs), min_ret_coefs))
    inequality_bounds = np.vstack((np.array(nonnegativity_constraints_bounds), min_ret_bound))
    inequality_coefs = cvxopt.matrix(inequality_coefs)
    inequality_bounds = cvxopt.matrix(inequality_bounds)
    return cvxopt.solvers.qp(
        risk_matrix, c,
        inequality_coefs, inequality_bounds,
        normalization_constraint_coefs, normalization_constraint_bound,
        options={"show_progress": False})


if __name__ == "__main__":
    EXPECTED_RETURN_WEIGHT = 0.0
    RISK_WEIGHT = 1.0 - EXPECTED_RETURN_WEIGHT
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    for i, c in enumerate(companies):
        c.expected_return = (i+1)/20
    # s = maximize_return_solver(companies)
    s = minimize_risk_solver(companies)
    # print(s.keys())
    # print(s["x"])
    # print(s["primal objective"])

    exp_ret_spread, risk_spread = exp_ret_risk_spreads(companies)
    w_s = weighted_sum_solver(
        companies, EXPECTED_RETURN_WEIGHT, RISK_WEIGHT,
        exp_ret_spread, risk_spread)
    weights = w_s["x"]
    risk = utils.portfolio_risk(companies, weights)
    exp_ret = utils.portfolio_expected_return(companies, weights)
    print(exp_ret, risk)

    e_c = epsilon_constrained_solver(
        companies, 0.95)
    e_weights = e_c["x"]
    risk = utils.portfolio_risk(companies, e_weights)
    exp_ret = utils.portfolio_expected_return(companies, e_weights)
    print(f"{e_c['primal objective']=}")
    print(exp_ret, risk)
    # print(e_c["x"])

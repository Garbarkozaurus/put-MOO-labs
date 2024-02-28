import numpy as np
import cvxopt

from company import Company
import data_loading
import utils


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
    normalization_constraint_coefs = cvxopt.matrix([[1.0] for _ in companies])
    # -1, because that's how >= is expressed
    nonnegativity_constraints_coefs = cvxopt.matrix(np.diagflat([-1.0 for _ in companies]))
    # every row in the argument is a column in the constraint matrix
    # representing the coefficients for variables in all constraints
    normalization_constraint_bound = cvxopt.matrix([1.0])
    nonnegativity_constraints_bounds = cvxopt.matrix([0.0] * len(companies))
    return cvxopt.solvers.lp(exp_ret_matrix, nonnegativity_constraints_coefs,
                             nonnegativity_constraints_bounds,
                             normalization_constraint_coefs,
                             normalization_constraint_bound)


def covariance_matrix_from_companies(
        companies: list[Company],
        history_len: int | None = None) -> np.ndarray[np.float64]:
    price_matrix = utils.company_list_to_price_matrix(companies, history_len)
    # keep the np.corrcoef() alternative in mind!
    return np.cov(price_matrix)


def minimize_risk_solver(companies: list[Company]):
    covariance_matrix = covariance_matrix_from_companies(companies)
    risk_matrix = cvxopt.matrix(covariance_matrix)

    # coefficients for the linear component of the optimized function
    c = cvxopt.matrix([0.0 for _ in companies])
    normalization_constraint_coefs = cvxopt.matrix([[1.0] for _ in companies])
    # -1, because that's how >= is expressed
    nonnegativity_constraints_coefs = cvxopt.matrix(np.diagflat([-1.0 for _ in companies]))
    # every row in the argument is a column in the constraint matrix
    # representing the coefficients for variables in all constraints
    normalization_constraint_bound = cvxopt.matrix([1.0])
    nonnegativity_constraints_bounds = cvxopt.matrix([0.0] * len(companies))
    return cvxopt.solvers.qp(risk_matrix, c, nonnegativity_constraints_coefs,
                             nonnegativity_constraints_bounds,
                             normalization_constraint_coefs,
                             normalization_constraint_bound)


def weighted_sum_solver(
        companies: list[Company],
        profit_weight: float,
        risk_weight: float,
        history_len: int | None = None) -> cvxopt.solvers.qp:
    """IMPORTANT: this function assumes that the values of expected return
    have already been calculated and stored in the expected_return
    field of appropriate companies"""
    # TODO
    # remember to normalize by dividing over the spread!
    pass


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    for i, c in enumerate(companies):
        c.expected_return = (i+1)/20
    s = maximize_return_solver(companies)
    # s = minimize_risk_solver(companies)
    print(s.keys())
    print(s["primal objective"])
    print(s["x"])

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
    normalization_constraint_coefs = [1.0 for _ in companies]
    # -1, because that's how >= is expressed
    nonnegativity_constraints_coefs = np.diagflat([-1.0 for _ in companies])
    # every row in the argument is a column in the constraint matrix
    # representing the coefficients for variables in all constraints
    constraint_coeffs = cvxopt.matrix(
        np.vstack((normalization_constraint_coefs,
                   nonnegativity_constraints_coefs)))
    # 1.0 for normalization, 0.0 for non-negativity
    constraint_bounds = cvxopt.matrix([1.0] + [0.0] * len(companies))
    return cvxopt.solvers.lp(exp_ret_matrix, constraint_coeffs, constraint_bounds)


def covariance_matrix_from_companies(
        companies: list[Company],
        history_len: int | None = None) -> np.ndarray[np.float64]:
    price_matrix = utils.company_list_to_price_matrix(companies, history_len)
    # keep the np.corrcoef() alternative in mind!
    return np.cov(price_matrix)


def minimize_risk_solver(companies: list[Company]):
    # TODO
    pass


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
    print(s.keys())
    print(s["primal objective"])
    print(s["x"])

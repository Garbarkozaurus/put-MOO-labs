import numpy as np
from typing import Iterable

from company import Company


def company_list_to_price_matrix(
        companies: list[Company],
        history_len: int | None = None) -> np.ndarray[np.float64]:
    """Given a list of companies returns a matrix containing all of their
    price histories.
    If `history_len` is None, the entire histories are included. Otherwise,
    only the `history_len` most recent prices are taken into account
    """
    if history_len is None:
        return np.array([company.prices for company in companies])
    return np.array([company.prices[-history_len:] for company in companies])


def covariance_matrix_from_companies(
        companies: list[Company],
        history_len: int | None = None) -> np.ndarray[np.float64]:
    price_matrix = company_list_to_price_matrix(companies, history_len)
    # keep the np.corrcoef() alternative in mind!
    return np.cov(price_matrix)


def portfolio_expected_return(
        companies: list[Company],
        weights: Iterable[float]) -> float:
    return sum([company.expected_return * weight
                for company, weight in zip(companies, weights)])


def portfolio_risk(
        companies: list[Company],
        weights: Iterable[float],
        history_len: int | None = None) -> float:
    weight_array = np.array(weights)
    cov_matrix = covariance_matrix_from_companies(companies, history_len)
    # multiplied by 0.5 for consistency with solver values
    return 0.5 * float(weight_array.T @ cov_matrix @ weight_array)


COMPANY_ORDER = [
    "SuperFuture", "Apples", "WorldNow", "Electronics123", "Photons",
    "SpaceNow", "PearPear", "PositiveCorrelation", "BetterTechnology",
    "ABCDE", "EnviroLike", "Moneymakers", "Fuel4", "MarsProject", "CPU-XYZ",
    "RoboticsX", "Lasers", "WaterForce", "SafeAndCare", "BetterTomorrow"]


def weights_in_valid_order(companies: list[Company],
                           weights: Iterable[float]) -> list[float]:
    """USE WITH CARE
    Creates a new list with the order of weight corresponding to the one given
    in the portfolio problem description"""
    weight_dict = dict(zip([c.name for c in companies], weights))
    new_weights = weights[:]
    for i, target_company in enumerate(COMPANY_ORDER):
        new_weights[i] = weight_dict[target_company]
    return new_weights


def export_front_to_file(companies: list[Company],
                         solutions: list[dict], export_path: str,
                         history_len: int | None = None) -> None:
    with open(export_path, "w+") as fp:
        for solution in solutions:
            weights = solution['x']
            expected_return = portfolio_expected_return(companies, weights)
            risk = portfolio_risk(companies, weights, history_len)
            fp.write(f"{expected_return} {risk}\n")

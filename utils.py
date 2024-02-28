import numpy as np

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

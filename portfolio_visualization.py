import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import Iterable


from company import Company
import data_loading
import problem_construction
import utils


def vis_portfolio_linear_regression_subplots(
        companies: list[Company], prediction_time: int,
        weights: Iterable[float],
        fitting_timeline_start: int = 0,
        plot_timeline_start: int = 0,
        num_rows: int = 4) -> None:
    n = len(companies[0].prices)
    x = np.array(list(range(n)))[fitting_timeline_start:].reshape((-1, 1))
    num_columns = int(np.ceil(len(companies)/num_rows))
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_columns)
    # fig.suptitle("Linear regression")
    for i, company in enumerate(companies):
        y = company.prices[fitting_timeline_start:]
        # model = Ridge(alpha=1e-3).fit(x, y)
        model = LinearRegression().fit(x, y)
        prediction_value = model.predict(np.array([[prediction_time]]))[0]
        pred_at_end = model.predict(np.array([[n-1]]))[0]
        row = i // num_columns
        column = i % num_columns
        ax[row, column].plot(company.prices)
        if fitting_timeline_start != 0:
            ax[row, column].axvline(fitting_timeline_start, color="red")
        ax[row, column].set_xlim([plot_timeline_start, prediction_time])
        ax[row, column].plot((n-1, prediction_time), (pred_at_end, prediction_value))
        ax[row, column].set_title(
            f"{company.name}: " +
            f"{np.round((prediction_value/y[-1]-1)*100, 2)}% " +
            f"{np.round(weights[i], 2)} | "
            f"{np.round((prediction_value/y[-1]-1) * weights[i], 2)}")
        ax[row, column].tick_params(
            axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1/")
    # solution = problem_construction.epsilon_constrained_solver(companies, min_ret=0.21790337174405583)
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.36, 0.0, 0.36, 0.0, 0.06, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0]
    vis_portfolio_linear_regression_subplots(companies, 200, weights)

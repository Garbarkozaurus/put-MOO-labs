import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from typing import Literal

from company import Company
import data_loading
import utils


def overview_linear_regression_iter(
        companies: list[Company], prediction_time: int,
        fitting_timeline_start: int = 0,
        plot_timeline_start: int = 0) -> None:
    n = len(companies[0].prices)
    x = np.array(list(range(n)))[fitting_timeline_start:].reshape((-1, 1))
    for company in companies:
        y = company.prices[fitting_timeline_start:]
        model = LinearRegression().fit(x, y)
        prediction_value = model.predict(np.array([[prediction_time]]))[0]
        pred_at_end = model.predict(np.array([[n-1]]))[0]
        plt.plot(company.prices)
        ax = plt.gca()
        ax.set_xlim([plot_timeline_start, prediction_time])
        plt.axvline(fitting_timeline_start, color="red")
        plt.plot((n-1, prediction_time), (pred_at_end, prediction_value))
        plt.title(f"{company.name} | Linear regression\n" +
                  f"Predicted price at {prediction_time}: {prediction_value} (return: {(prediction_value/y[-1]-1)*100}%)" +
                  f" score: {model.score(x, y)}")
        plt.show()


def overview_linear_regression_subplots(
        companies: list[Company], prediction_time: int,
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
            f"{np.round((prediction_value/y[-1]-1)*100, 2)}% | {np.round(prediction_value, 3)}" +
            f" | {np.round(model.score(x, y), 3)}")
        ax[row, column].tick_params(
            axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()


def overview_polynomial_subplots(
        companies: list[Company],
        prediction_time: int,
        degree: int,
        fitting_timeline_start: int = 0,
        plot_timeline_start: int = 0,
        num_rows: int = 4) -> None:
    n = len(companies[0].prices)
    x = np.array(list(range(n)))[fitting_timeline_start:].reshape((-1, 1))
    num_columns = int(np.ceil(len(companies)/num_rows))
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_columns)
    x_plot = list(range(n-1, prediction_time+1))
    # fig.suptitle("Linear regression")
    for i, company in enumerate(companies):
        y = company.prices[fitting_timeline_start:]
        # model = Ridge(alpha=1e-3).fit(x, y)
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1)).fit(x, y)
        # model = make_pipeline(SplineTransformer(n_knots=5, degree=degree), Ridge(alpha=1e-3)).fit(x, y)
        prediction_value = model.predict(np.array([[prediction_time]]))[0]
        row = i // num_columns
        column = i % num_columns
        ax[row, column].plot(company.prices)
        if fitting_timeline_start != 0:
            ax[row, column].axvline(fitting_timeline_start, color="red")
        ax[row, column].set_xlim([plot_timeline_start, prediction_time])
        ax[row, column].plot(x_plot, model.predict(np.array([x_plot]).reshape((-1, 1))))
        # new_x = list(range(prediction_time))
        # ax[row, column].plot(new_x, model.predict(np.array([new_x]).reshape((-1, 1))))
        ax[row, column].set_title(
            f"{company.name}: " +
            f"{np.round((prediction_value/y[-1]-1)*100, 2)}% | {np.round(prediction_value, 3)}" +
            f" | {np.round(model.score(x, y), 3)}")
        ax[row, column].tick_params(
            axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()


def predict_expected_return_linear_regression(
        company: Company,
        prediction_time: int,
        fitting_timeline_start: int = 0,
        mode: Literal["Value"] | Literal["increase"] = "increase") -> tuple[float, float]:
    """Returns pair: predicted expected return, 'prediction confidence' (model.score)
    """
    n = len(company.prices)
    x = np.array(list(range(n)))[fitting_timeline_start:].reshape((-1, 1))
    y = company.prices[fitting_timeline_start:]
    model = LinearRegression().fit(x, y)
    prediction_value = model.predict(np.array([[prediction_time]]))[0]
    score = model.score(x, y)
    if mode == "increase":
        return (prediction_value/y[-1])-1, score
    return prediction_value, score


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1")
    overview_linear_regression_subplots(companies, 200)
    overview_polynomial_subplots(companies, 200, 2)
    print(predict_expected_return_linear_regression(companies[0], 200))

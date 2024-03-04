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
import return_estimation


def vis_portfolio_linear_regression_subplots(
        companies: list[Company], prediction_time: int,
        weights: Iterable[float],
        fitting_timeline_start: int = 0,
        plot_timeline_start: int = 0,
        num_rows: int = 4,
        export_pdf: bool = False,
        pdf_title: str = "portfolio.pdf") -> None:
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
    if export_pdf:
        plt.gcf().set_size_inches(25 * 0.7, 18 * 0.6)
        plt.savefig(pdf_title, format="pdf")
    plt.show()


if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle1/")
    weights_portfolio1 = [8.644665649845455e-07, 3.2298170638724315e-08, 1.3824240077869665e-08, 2.1611643635980332e-08, 1.8732855958022932e-06, 1.8413226578557183e-08, 1.8843236108265335e-09, 0.0004081023422847454, 0.024427478059861403, 0.3660731075758385, 2.987919062764484e-08, 0.3668022084828672, 4.23305172698446e-08, 0.061618815871245035, 0.18066607274170104, 3.706977748432357e-08, 6.306775209173726e-08, 1.0251652094173745e-08, 5.559817292282156e-07, 6.505618178746683e-07]
    vis_portfolio_linear_regression_subplots(companies, 200, weights_portfolio1)
    for company in companies:
        company.expected_return = return_estimation.predict_expected_return_linear_regression(company, 200)[0]

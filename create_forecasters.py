import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
import pandas as pd
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

import data_loading

if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle3/")
    x = np.array(list(range(len(companies[0].prices))))
    for c in companies:
        y = c.prices
        forecaster = ForecasterAutoregDirect(
                    regressor = RandomForestRegressor(
                        random_state=0, max_depth = 10,
                        n_estimators=500),
                    steps=100,
                    lags = list(range(1, 51))
                )
        forecaster.fit(pd.Series(y))
        save_forecaster(forecaster, f"{c.name}_bundle3.joblib")
    # for i, c in enumerate(companies):
    #     print(f"Processing {i+1}/{len(companies)} - {c.name}")
    #     forecaster_path = f"./saved_forecasters/{c.name}_bundle3.joblib"
    #     imported_forecaster = load_forecaster(forecaster_path)
    #     forecast_path = f"./saved_forecasts/bundle3/{c.name}.csv"
    #     pd.DataFrame(imported_forecaster.predict(100)).to_csv(forecast_path, sep=',', header=False)

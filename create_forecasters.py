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

import data_loading

if __name__ == "__main__":
    companies = data_loading.load_all_companies_from_dir("./data/Bundle2/")
    x = np.array(list(range(201)))
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
        save_forecaster(forecaster, f"{c.name}_bundle2.joblib")

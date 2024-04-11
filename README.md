# CElegans-Stimulus-Data

## Install

```
pip install -r requirements.txt
```
Python 3.9 was used.

## Run

```
python3 analysis.py [--model_type <NONE/VAR/VARMA>] [--data_type <INPUT/OUTPUT>] [--transformation <NONE/Z/LOG/SQRT/BOX-COX>]
```

`analysis.py` contains the transformations applied to the data. Arguments:

- `model_type` is the model to be fitted. Since the data does not follow a Gaussian value distribution, only `NONE` can be applied;
- `data_type` states the data to be analyzed, either input-only or output-only;
- `transformation` is a transformation applied to the data, with the objective of approximating it to a Gaussian distribution. `Z` performs a standardization (Z-score), `LOG` the natural logarithm, `SQRT` the square root, and `BOX-COX` applies the Box-Cox method from Scipy.stats.

`output` directory contains folders named `<data_type>_<transformation>`, where the results for this configuration are stored. To recompute the output, the folder must be deleted before running the code.

Inside each mentioned folder, the folder: `acf` contains the Autocorrelation graphs, `adf_tests` the Augmented Dickey-Fuller (stationarity) tests, `histograms` the value histogram graphs, `pacf` the Partial Autocorrelation graphs, `qqplots` the Q-Q plots (visual check for Gaussian distribution), `shapiro_tests` the Shapiro-Wilk tests (Gaussian distribution tests).

Note: ADF tests the null hypothesis of a unit root present in the series, if this hypothesis is true (high p-value), then the series is non-stationary. Shapiro-Wilk tests the null hypothesis of the data following a Gaussian distribution, if this hypothesis is true (high p-value), then the values of the series are Gaussian distributed.

All `result.csv`, `analysis2.py` and `report.html` can be ignored, they were an attempt to analyze the data in another way.

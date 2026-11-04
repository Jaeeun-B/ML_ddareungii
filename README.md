# Ttareungyi Demand Forecasting

따릉이 대여량 예측 · EURON 8th ML project, summer 2025

Forecasting daily rentals of Seoul's public bike-sharing service, Ttareungyi, for four districts in eastern Seoul: Gwangjin, Dongdaemun, Seongdong and Jungnang. Each team member built base models on their own branch, and the final 2022 forecast stacks their predictions with an XGBoost meta-model.

## Folders

| Folder | What's inside | By |
|---|---|---|
| `lightgbm/` | LightGBM with feature engineering and tuning, 2021 and 2022 predictions | [@Jaeeun-B](https://github.com/Jaeeun-B) |
| `catboost-xgboost-stacking/` | CatBoost base model and the XGBoost stacking ensemble | [@jis-25](https://github.com/jis-25) |
| `eda-prophet-knn/` | Exploratory data analysis, Prophet and KNN models | [@yeone0511](https://github.com/yeone0511) |
| `arima-sarima-rf/` | ARIMA, SARIMA and Random Forest models | [@yunjijeong0](https://github.com/yunjijeong0) |

Each folder was merged from its author's branch, so the commit history and authorship are preserved.

## Setup

- Target: daily rentals per district, in thousands. Gwangjin averages about 6,200 rentals a day.
- Metrics: RMSE and MAE per district.
- Base models produce predictions for 2021, which train the stacking meta-model, and for 2022, which form the final forecast.

## A leak we caught

The first LightGBM runs scored an RMSE below 0.4 in every district, which was too good to be true. The features included the other districts' rentals for the same day, information that would not exist at forecast time. Once those columns were removed, errors rose to a realistic level, and every later experiment started from that baseline.

## LightGBM: what moved the error

RMSE in thousands of rentals per day, on a random 20% hold-out of the training data.

| Step | Gwangjin | Dongdaemun | Seongdong | Jungnang |
|---|---:|---:|---:|---:|
| Date features only, leak removed | 1.92 | 1.26 | 1.62 | 0.95 |
| + holiday and weekend flags | 1.89 | 1.24 | 1.58 | 0.96 |
| + cyclic encoding of month and weekday | 1.84 | 1.22 | 1.55 | 0.94 |
| + lag and rolling-mean features | 1.35 | 0.89 | 1.20 | 0.69 |
| + Optuna hyperparameter tuning | 1.28 | 0.86 | 1.16 | 0.69 |
| + Optuna search over lag and rolling windows | 1.18 | 0.83 | 1.08 | 0.54 |

Holiday flags helped three districts but not Jungnang. Lag and rolling features made the largest single difference.

**Limitations.** The hold-out is a random split, so a validation day's neighbouring days can sit in the training set and feed its lag features. That makes the last three rows optimistic compared with true day-ahead forecasting, and the Optuna rows were tuned and scored on the same validation split. A time-ordered split, for example training through 2020 and validating on 2021, would give a fairer estimate.

## Ensemble

`catboost-xgboost-stacking/` combines the 2021 predictions of LightGBM, CatBoost, Random Forest and KNN with an XGBoost meta-model, then applies it to the 2022 base predictions. Prophet, ARIMA and SARIMA were also tried as base models.

## Running the notebooks

The notebooks were run on Google Colab and locally with pandas, scikit-learn, LightGBM, CatBoost, XGBoost, Optuna, Prophet and statsmodels. The raw `train.csv` is not included in this repository, and file paths inside the notebooks point to each author's own drive.

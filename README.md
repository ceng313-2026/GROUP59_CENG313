# GROUP59_CENG313 — Hull Tactical Market Prediction

Course project for **CENG313 – Introduction to Data Science**.

## Team

| Name | Student ID | GitHub |
|------|------------|--------|
| Sadık Onur Hayta | 23118080046 | [@onurhayta](https://github.com/onurhayta) |

- **Group ID:** 59
- **Competition level:** Light (1 student)
- **Modality:** Time series

## Competition

- **Kaggle competition:** [Hull Tactical – Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- **Task:** For each trading day, choose an allocation to the S&P 500 in the range **[0, 2]**
  (0 = fully in cash / risk-free asset, 1 = fully invested, 2 = 2x leveraged). The rest of the
  capital earns the risk-free rate.
- **Evaluation metric:** A modified (volatility-adjusted) Sharpe ratio. The score is penalised when
  the strategy's volatility exceeds **1.2×** the market's volatility, and when the strategy's return
  falls below the market's return. Re-implemented in `src/utils.py` and verified against the official
  scoring code (Kaggle metric `hull-competition-sharpe`): identical scores on 1,000 random test cases.
- **Competition format:** Code competition (predictions are served through Kaggle's evaluation API).
  Timeline: started 15 Sep 2025, final submission deadline 15 Dec 2025, ended 16 Jun 2026.

## Dataset

`train.csv` contains **9,048 trading days** (`date_id` 0–9047, ≈36 years) and **98 columns**:
`date_id`, 94 anonymised features and 3 target columns. Dates are anonymised, so `date_id` only gives the order of the days.

| Prefix | Feature group | # features |
|--------|---------------|-----------:|
| `M*`   | Market dynamics / technical | 18 |
| `E*`   | Macro-economic | 20 |
| `I*`   | Interest rates | 9 |
| `P*`   | Price / valuation | 13 |
| `V*`   | Volatility | 13 |
| `S*`   | Sentiment | 12 |
| `D*`   | Binary / dummy indicators | 9 |

Target columns:

- `forward_returns` – next-day S&P 500 return
- `risk_free_rate` – federal funds rate
- `market_forward_excess_returns` – forward return relative to expectations (demeaned with a long-term rolling average and clipped); used as the prediction target

`test.csv` has the same features plus lagged target columns and an `is_scored` flag.

> The data files are **not** stored in this repository (see `.gitignore`) because the competition rules do not allow redistributing the data.
> They are downloaded from Kaggle inside the notebook (see *How to run*).

## Repository structure

```
group59_CENG313/
├── README.md
├── requirements.txt
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA + baseline models
├── assignment2_solution.ipynb       # Assignment 2: proposed solution (to be added)
└── src/
    ├── __init__.py
    ├── data.py      # data loading, feature groups, missing-value summary, walk-forward splits
    ├── models.py    # model interface, constant / Ridge / LightGBM baselines, walk-forward evaluation
    └── utils.py     # competition metric and evaluation helpers
```

## Approach

### Assignment 1 – EDA and baselines ✅

**Exploratory data analysis** (`assignment1_eda_baseline.ipynb`, Section 2):

- **Missing values:** for the first 1,006 days only the 9 `D*` features exist; most other features start at day 1006 and some much later (e.g. `E7` ≈ day 6969). Missingness is structural, not random.
- **Target distribution:** daily mean 0.047% and std 1.05% (≈12% annual return, ≈17% annual volatility); fat tails (excess kurtosis ≈ 2.2), slight negative skew, clipped at about ±4%. Volatility varies between ≈9% and ≈34% per year.
- **Predictability:** returns have almost no autocorrelation, but absolute returns are strongly autocorrelated, i.e. **direction is hard to predict, volatility is predictable**.
- **Feature–target relationships:** all correlations are weak (|Spearman| ≤ 0.054), noisy and non-linear; many features are highly correlated with each other.
- **Stability:** the relationships change over time and nearly vanish in the most recent period (non-stationarity).

**Validation:** expanding-window **walk-forward** validation on rows from `date_id` 1006 onwards, with 8 consecutive validation folds of 504 days (≈2 years) and a 1-day gap. Each fold is scored with the competition metric.

**Baselines** (`src/models.py`):

1. Constant allocation 0 (cash), 1 (buy-and-hold) and 2 (2× leverage)
2. Ridge regression (median imputation + standardisation, α = 100)
3. LightGBM

Ridge and LightGBM predict `market_forward_excess_returns`; the prediction is mapped to an allocation with
`clip(1 + 0.5 · prediction / std_train, 0, 2)`.

### Assignment 2 – Proposed solution *(planned)*

Based on the EDA: volatility-aware position sizing, a tuned prediction-to-allocation mapping, feature selection and engineering, handling non-stationarity (rolling vs. expanding windows) and ensembling. Compared against the Assignment 1 baselines on the same folds.

## Results

Walk-forward validation, 8 folds × 504 days (`date_id` 5016–9047):

| Model | Mean fold score | Std across folds | Full-period score | Folds beating buy-and-hold |
|-------|----------------:|-----------------:|------------------:|---------------------------:|
| Constant 1 (buy-and-hold) | **0.844** | 0.590 | **0.766** | – |
| LightGBM | 0.718 | 0.365 | 0.699 | 5 / 8 |
| Ridge | 0.519 | 0.343 | 0.520 | 2 / 8 |
| Constant 2 (2× leverage) | 0.431 | 0.322 | 0.381 | 1 / 8 |
| Constant 0 (cash) | 0.000 | 0.000 | 0.000 | 1 / 8 |

Buy-and-hold is not beaten overall. LightGBM is the strongest and most stable model; Ridge is penalised for exceeding the volatility limit (strategy volatility 23.6% vs. market 15.9%).

## How to run

The notebooks are designed for **Google Colab**.

1. Join the competition on Kaggle and accept the [competition rules](https://www.kaggle.com/competitions/hull-tactical-market-prediction/rules) (required to download the data).
2. Create a Kaggle API token (*Kaggle → Settings → API Tokens → Generate*).
3. Open the notebook in Colab (use the *Open in Colab* badge at the top of the notebook, or *File → Open notebook → GitHub*).
4. In Colab, open **Secrets** (key icon on the left), add a secret named `KAGGLE_API_TOKEN` with your token, and enable notebook access.
5. Run all cells. The setup cells clone this repository (so that `src/` can be imported) and download the data with `kagglehub`.

Locally:

```bash
git clone https://github.com/onurhayta/group59_CENG313.git
cd group59_CENG313
pip install -r requirements.txt kagglehub
export KAGGLE_API_TOKEN=<your token>
jupyter notebook assignment1_eda_baseline.ipynb
```

## Progress log

| Date | Work done |
|------|-----------|
| 2026-10-29 | Repository created; README, `.gitignore`, `requirements.txt`, `src` package and `src/data.py` (data loading, feature groups, walk-forward splits) added. |
| 2026-11-05 | Competition metric (`src/utils.py`) and baseline models (`src/models.py`) added. Assignment 1 notebook: data download, EDA (missing values, target distribution, volatility, feature–target relationships, stability), walk-forward validation and baseline results. README updated with results. |
| 2026-11-07 | Competition metric verified against the official Kaggle scoring code (identical scores on 1,000 random cases); zero-volatility handling aligned with the official code. |

## Kaggle submission history

The competition closed before this project started (entry deadline 8 Dec 2025, final submission deadline 15 Dec 2025), so no submissions could be made to the leaderboard. All scores above come from our own walk-forward validation using a re-implementation of the competition metric.

| Date | Model / notebook version | Public score | Private score | Notes |
|------|--------------------------|--------------|---------------|-------|
| – | – | – | – | Competition closed; see local validation results above |
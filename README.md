<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Course Project Description

- **Dataset:** [Kaggle - Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)


### Course Deliverables

#### PR #1

- Azure Databricks Environment Setup
- Select dataset [Kaggle - Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- Run python notebooks in databricks cluster for fraud_credit_cards usecase
- Create "DataProcessor" and "FraudModel" classes
- Push data.csv to databricks volume
- Push package.whl to databricks volume
- Create main.py to preprocess data, train model, and evaluate model
- Fix pre-commit checks


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

### To create a new environment and create a lockfile, run:

```
uv venv -p 3.11.11 .venv
source .venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```

### To Build fraud_credit_cards package

```
uv build
```

To install and run fraud_credit_cards package

```
uv pip install dist/fraud_credit_cards-0.0.1-py3-none-any.whl
```

```
uv run python main.py
```

### Pre-Commit Checks

To run pre-commit checks

```
uv run pre-commit run --all-files
```

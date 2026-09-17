# Credit Card Propensity & Decision Intelligence Platform

A machine learning and decision-intelligence project for estimating credit-card default risk and translating model predictions into actionable risk prioritization.

## Project Overview

This project explores how historical customer and credit-card behavior can be used to estimate the probability of a customer defaulting on their credit-card payment in the subsequent month.

The project goes beyond building a binary classification model. The objective is to build a decision-oriented analytical system that can:

1. Estimate customer default probability.
2. Rank customers according to relative risk.
3. Group customers into meaningful risk segments.
4. Incorporate business considerations such as potential financial exposure, false-positive/false-negative costs, intervention costs, and operational capacity.
5. Translate model outputs into risk-management priorities.

The project is intended as a prototype decision-intelligence system, not as a production banking or automated credit-approval system.

## Business Problem

Credit-risk teams have limited resources and cannot necessarily provide the same level of attention to every customer.

The central question of this project is:

> Which customers should receive elevated risk-management attention given their estimated probability of default and the constraints of the business?

The primary intended user is a:

- Credit Risk Analyst
- Credit Risk Manager

The system is designed around:

```text
Customer historical data
        ↓
Data validation
        ↓
Data understanding & exploration
        ↓
Feature engineering
        ↓
Default probability model
        ↓
Risk probability
        ↓
Customer ranking / segmentation
        ↓
Business constraints & costs
        ↓
Risk-management priority
```

## Prediction Problem

The underlying machine-learning target is binary:

- `0` → customer does not default in the subsequent month
- `1` → customer defaults in the subsequent month

The model's primary output will be:

```text
P(default | available customer information)
```

Risk bands will be constructed at the decision layer from predicted probability and business considerations rather than treated as separate machine-learning classes.

## Decision Intelligence

The model prediction is only one component of the system.

### Prediction

> How likely is this customer to default?

### Risk prioritization

> How does this customer's estimated risk compare with other customers?

### Decision

> Given risk, financial exposure, intervention cost, and available operational capacity, which customers warrant additional attention?

Potential risk-management actions may include:

- payment reminders
- temporary spending restrictions
- structured repayment options
- additional risk-management attention

The dataset does not contain intervention outcomes or causal evidence, so this project will not claim that these interventions prevent default.

## Dataset

This project uses the **UCI Default of Credit Card Clients Dataset**.

Official source:

https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

The dataset contains historical credit-card customer information, including:

- demographic information
- credit exposure
- repayment-status history
- billing/statement behavior
- payment behavior

The dataset contains approximately 30,000 customer observations and six months of historical information.

The raw dataset is not committed to this repository. Users should obtain it from the official UCI source and place the raw file under:

```text
data/raw/
```

## Temporal Interpretation

The dataset contains historical information covering six months.

For this project, the modeling interpretation is:

```text
Historical information
(April–September 2005)
        ↓
Estimate probability of
subsequent-month default
```

The exact operational prediction timestamp is not available from the dataset documentation and is therefore treated as a modeling assumption.

## Analytical Boundaries

### Historical population

The dataset represents a specific historical population and should not automatically be treated as representative of modern banking customers or customers in other countries or markets.

### Limited available information

Real-world financial institutions typically have access to substantially more information than is available in this dataset.

### Prediction is not causation

A relationship between customer behavior and default probability does not demonstrate that a particular intervention will prevent default.

### Intervention effectiveness

The dataset does not provide information about intervention cost, intervention effectiveness, customer response to intervention, or financial recovery after intervention.

Any business-cost or intervention simulation in this project will therefore be explicitly identified as an assumption-based prototype.

### Production use

This project is not intended to be:

- a production banking system
- an automated credit approval/rejection system
- a regulatory-compliant credit decision system
- evidence that a particular intervention will prevent default

A real-world deployment would require additional data, validation, monitoring, governance, fairness analysis, regulatory considerations, retraining, and potentially architectural changes.

## Repository Structure

```text
credit-card-propensity-and-decision-intelligence-platform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
├── models/
├── notebooks/
├── reports/
│
├── src/
│   └── propensity_prediction/
│       ├── data/
│       └── ...
│
├── tests/
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Directory responsibilities

#### `data/raw/`

Original datasets obtained from external sources. Raw source data is not committed to Git.

#### `data/processed/`

Meaningful processed datasets produced by the project pipeline. These may be committed when useful for reproducibility and inspection.

#### `notebooks/`

Exploratory analysis, investigation, visualization, and experimentation.

#### `src/`

Reusable Python project code, including data loading, validation, preprocessing, feature engineering, modeling, evaluation, and decision-intelligence components.

#### `models/`

Trained model artifacts and related model outputs.

#### `reports/`

Meaningful analytical outputs such as visualizations, evaluation reports, and business-analysis results.

#### `tests/`

Automated tests for reusable project functionality.

#### `docs/`

Project documentation, architectural decisions, methodology, and supporting documentation.

## Environment Setup

### Requirements

The project is developed using Python.

Create a virtual environment in the project root:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Install the project package in editable mode:

```bash
pip install -e .
```

## Running Tests

Run the complete test suite with:

```bash
pytest tests -v
```

## Dependency Management

Python dependencies are recorded in `requirements.txt`.

The development environment is version-pinned using:

```bash
pip freeze > requirements.txt
```

The project package itself is configured through `pyproject.toml`.

## Current Project Status

### Phase 0 — Problem Definition

**Completed**

Established:

- business problem
- target variable
- intended user
- prediction objective
- risk representation
- business decision framing
- analytical boundaries
- dataset selection

### Phase 1 — Project & Data Foundation

**In progress / near completion**

Completed components include:

- repository structure
- virtual environment
- dependency setup
- raw dataset acquisition
- raw dataset loading
- initial data validation

### Phase 2 — Data Understanding & Exploration

**Next**

Planned work includes:

- understanding dataset semantics
- examining distributions
- understanding repayment behavior
- investigating relationships with default
- identifying data-quality and domain considerations
- developing an analytical understanding before modeling

## Modeling Principles

The project will prioritize analytical correctness over simply achieving a high accuracy score.

Important considerations include:

- train/validation/test separation
- leakage prevention
- class imbalance
- appropriate baseline models
- probability calibration
- discrimination metrics
- threshold selection
- ranking quality
- false-positive and false-negative costs
- financial exposure
- operational capacity
- model interpretability
- reproducibility

Accuracy alone will not be treated as sufficient evidence of model usefulness.

## Risk Prioritization Strategy

The intended decision layer combines:

```text
Predicted probability
        +
Relative customer ranking
        +
Financial exposure
        +
False-positive / false-negative costs
        +
Intervention cost
        +
Operational capacity
        ↓
Risk-management priority
```

This allows the project to distinguish between:

> Who has the highest predicted probability of default?

and:

> Who should receive limited risk-management resources?

These are related but different questions.

## Future Architecture

The project is expected to evolve toward:

```text
Customer Data
     ↓
Data Ingestion
     ↓
Validation
     ↓
Cleaning / Transformation
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Machine Learning
     ↓
Model Evaluation
     ↓
Probability / Risk Estimation
     ↓
Explainability
     ↓
Customer Segmentation
     ↓
Decision Intelligence
     ↓
Analytics Interface / API
     ↓
Optional GenAI Interface
```

GenAI, if introduced, will sit above deterministic analytical and decision tools. It will not replace the underlying statistical and business logic.

## Technology Stack

Current technologies:

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter
- Pytest

Additional technologies will be introduced only when justified by project requirements.

## Development Philosophy

This project emphasizes:

- reproducibility
- analytical reasoning
- explicit assumptions
- separation of concerns
- testable code
- data-quality validation
- business-aware modeling
- explainability
- responsible interpretation of model outputs

The objective is not simply to train a model.

The objective is to demonstrate how **data, machine learning, and business reasoning can work together to support decisions**.

## License & Dataset Attribution

The dataset is provided by the UCI Machine Learning Repository.

Refer to the official dataset page for the dataset's licensing and attribution requirements:

https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients
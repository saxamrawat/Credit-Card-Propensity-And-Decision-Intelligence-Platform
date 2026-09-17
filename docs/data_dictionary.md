# Data Dictionary

## Dataset

**UCI Default of Credit Card Clients Dataset**

Source:

https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

The dataset contains information about credit-card clients and whether they defaulted on their payment in the subsequent month.

The original dataset contains 23 explanatory variables and one binary target variable. The project preserves the original UCI variable identifiers alongside their descriptive names during the raw-loading stage.

---

## Variable Groups

The variables can be organized into the following conceptual groups:

1. Customer identifier
2. Credit exposure
3. Demographic information
4. Repayment-status history
5. Billing / statement history
6. Payment history
7. Target

---

## Variables

| UCI ID | Variable | Description | Role |
|---|---|---|---|
| X1 | `LIMIT_BAL` | Amount of the given credit, including individual consumer credit and their family/supplementary credit | Feature |
| X2 | `SEX` | Gender of the client | Feature |
| X3 | `EDUCATION` | Education level of the client | Feature |
| X4 | `MARRIAGE` | Marital status of the client | Feature |
| X5 | `AGE` | Age of the client in years | Feature |
| X6 | `PAY_0` | Repayment status in September 2005 | Feature |
| X7 | `PAY_2` | Repayment status in August 2005 | Feature |
| X8 | `PAY_3` | Repayment status in July 2005 | Feature |
| X9 | `PAY_4` | Repayment status in June 2005 | Feature |
| X10 | `PAY_5` | Repayment status in May 2005 | Feature |
| X11 | `PAY_6` | Repayment status in April 2005 | Feature |
| X12 | `BILL_AMT1` | Amount of bill statement in September 2005 | Feature |
| X13 | `BILL_AMT2` | Amount of bill statement in August 2005 | Feature |
| X14 | `BILL_AMT3` | Amount of bill statement in July 2005 | Feature |
| X15 | `BILL_AMT4` | Amount of bill statement in June 2005 | Feature |
| X16 | `BILL_AMT5` | Amount of bill statement in May 2005 | Feature |
| X17 | `BILL_AMT6` | Amount of bill statement in April 2005 | Feature |
| X18 | `PAY_AMT1` | Amount of previous payment in September 2005 | Feature |
| X19 | `PAY_AMT2` | Amount of previous payment in August 2005 | Feature |
| X20 | `PAY_AMT3` | Amount of previous payment in July 2005 | Feature |
| X21 | `PAY_AMT4` | Amount of previous payment in June 2005 | Feature |
| X22 | `PAY_AMT5` | Amount of previous payment in May 2005 | Feature |
| X23 | `PAY_AMT6` | Amount of previous payment in April 2005 | Feature |
| Y | `default payment next month` | Whether the client defaulted on payment in the subsequent month | Target |

---

## Customer Identifier

| UCI ID | Variable | Description |
|---|---|---|
| — | `ID` | Client identifier |

`ID` identifies the customer observation and is not intended to be used as a predictive feature.

The identifier should be retained long enough to support traceability and validation, but excluded from model features.

---

## Demographic Variables

### SEX

The dataset documentation defines:

| Value | Meaning |
|---:|---|
| 1 | Male |
| 2 | Female |

### EDUCATION

The documented categories are:

| Value | Meaning |
|---:|---|
| 1 | Graduate school |
| 2 | University |
| 3 | High school |
| 4 | Others |

The validation stage identified additional observed values `0`, `5`, and `6`.

These values are outside the documented category definitions and are therefore treated as **data-quality findings requiring investigation**, rather than being assigned meanings without evidence.

### MARRIAGE

The documented categories are:

| Value | Meaning |
|---:|---|
| 1 | Married |
| 2 | Single |
| 3 | Others |

The validation stage identified an observed value of `0`, which is outside the documented category definitions.

The project will not assign an interpretation to this value until a cleaning decision is explicitly justified.

---

## Repayment Status Variables

The repayment-status variables describe the client's repayment status for each month:

```text
PAY_0 → September 2005
PAY_2 → August 2005
PAY_3 → July 2005
PAY_4 → June 2005
PAY_5 → May 2005
PAY_6 → April 2005
```

The dataset documentation describes these variables in terms of months of delay in payment.

The exact encoding of the repayment-status categories will be investigated during exploratory analysis and preprocessing rather than assumed solely from the integer storage type.

---

## Billing / Statement Variables

The billing variables represent statement amounts:

```text
BILL_AMT1 → September 2005
BILL_AMT2 → August 2005
BILL_AMT3 → July 2005
BILL_AMT4 → June 2005
BILL_AMT5 → May 2005
BILL_AMT6 → April 2005
```

These variables represent billing/statement behavior and should not be described simply as "principal remaining."

---

## Payment Variables

The previous-payment variables represent payment amounts:

```text
PAY_AMT1 → September 2005
PAY_AMT2 → August 2005
PAY_AMT3 → July 2005
PAY_AMT4 → June 2005
PAY_AMT5 → May 2005
PAY_AMT6 → April 2005
```

These variables describe historical payment behavior.

---

## Target Variable

### `default payment next month`

Binary target:

| Value | Meaning |
|---:|---|
| 0 | No default |
| 1 | Default |

The machine-learning model will use this as the prediction target.

The model will ultimately estimate:

```text
P(default | available historical customer information)
```

Risk categories will be constructed later at the decision layer and will not replace the binary target.

---

## Temporal Structure

The historical variables cover April through September 2005.

The temporal ordering is:

```text
April 2005
    ↓
May 2005
    ↓
June 2005
    ↓
July 2005
    ↓
August 2005
    ↓
September 2005
    ↓
Subsequent-month default target
```

The exact operational prediction timestamp is not specified sufficiently by the dataset documentation. Therefore, the project treats the subsequent-month prediction interpretation as a modeling assumption.

Temporal availability will be considered explicitly during leakage analysis.

---

## Modeling Roles

### Identifier

```text
ID
```

Used for identification and traceability, not model prediction.

### Features

```text
LIMIT_BAL
SEX
EDUCATION
MARRIAGE
AGE
PAY_0 ... PAY_6
BILL_AMT1 ... BILL_AMT6
PAY_AMT1 ... PAY_AMT6
```

### Target

```text
default payment next month
```

---

## Important Data-Quality Findings

Initial validation established:

- 30,000 observations
- 25 columns in the loaded MultiIndex representation
- no missing values
- no duplicate complete rows
- target values restricted to `0` and `1`
- `ID` is being validated separately for uniqueness
- unexpected `EDUCATION` values: `0`, `5`, `6`
- unexpected `MARRIAGE` value: `0`
- no negative credit-limit values
- no negative previous-payment amounts
- age range observed: 21–79

These findings describe the raw data and do not represent preprocessing decisions.

---

## Data Interpretation Principles

The following principles will be maintained throughout the project:

1. Integer storage does not automatically imply a continuous numerical variable.
2. A value outside an assumed range is a validation finding before it is a cleaning decision.
3. Historical variables must be interpreted according to their temporal ordering.
4. Influence on the target does not by itself constitute data leakage; availability at prediction time is the key consideration.
5. The customer identifier should not be used as a predictive feature.
6. Raw data should remain unchanged; cleaning and transformation occur in later stages.
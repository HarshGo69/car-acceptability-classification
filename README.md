# Car Acceptability Classification

A machine learning project that predicts a car's acceptability rating (unacceptable, acceptable, good, very good) from its buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety rating.

## Dataset
[UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation) — 1,728 records, 6 categorical features, 4-class target.

## Approach
- Exploratory data analysis on feature distributions and class balance
- Label encoding to convert categorical text features into numeric form
- Feature scaling with StandardScaler
- Stratified train/test split to preserve class ratios
- Trained and compared two models: Logistic Regression and Random Forest
- Evaluated using accuracy, precision, recall, F1 (weighted, since this is multi-class)
- 5-fold cross-validation for a more reliable performance estimate
- Built a scikit-learn Pipeline chaining preprocessing and modeling
- Visualized class distribution, safety vs class, feature importance, and confusion matrix

## Results
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.688 | 0.575 | 0.688 | 0.609 |
| Random Forest | 0.983 | 0.983 | 0.983 | 0.983 |

5-fold cross-validation average accuracy: 0.815

## Key finding
Random Forest substantially outperformed Logistic Regression, likely because the relationship between features like safety and price tiers and the acceptability class isn't linear, something tree-based models capture better. Safety (0.28) and passenger capacity (0.22) were the strongest predictors, while doors (0.07) had the least influence.

## Tools
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## How to run
1. Clone this repo
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open the notebook in Jupyter and run all cells

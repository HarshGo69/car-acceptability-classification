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
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | X.XX | X.XX |
| Random Forest | X.XX | X.XX |

Cross-validation average accuracy: X.XX

## Key finding
Safety and passenger capacity were the strongest predictors of acceptability, while doors and luggage boot size had comparatively less influence.

## Tools
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## How to run
1. Clone this repo
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `Scikeit_learn.ipynb` in Jupyter and run all cells

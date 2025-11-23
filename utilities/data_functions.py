import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import (t, sem, ttest_ind, randint, uniform, norm, loguniform)
import numpy as np
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             average_precision_score)

risk_score_weighted = [
    ("age", lambda r: r["age"] >= 66, 3),
    ("hypertension", lambda r: r["hypertension"] == 1, 2),
    ("heart_disease", lambda r: r["heart_disease"] == 1, 2),
    ("bmi", lambda r: r["bmi"] >= 29, 1),
    ("glucose", lambda r: r["avg_glucose_level"] >= 124, 1),
    ("smoking_former", lambda r: r["smoking_status"] == "formerly_smoked", 2),
    ("smoking_current", lambda r: r["smoking_status"] == "smokes", 1),
    ("ever_married", lambda r: r["ever_married"] == 1, 1),
    ("self_employed", lambda r: r["work_type"] == "self_employed", 1),
]


def outliers_iqr(data: pd.DataFrame) -> None:
    """
    Detects and prints the number of outliers in each numeric column of a DataFrame using the IQR method.
    """
    data = pd.DataFrame(data)

    outlier_counts = {}
    for column in data.select_dtypes(include="number"):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        
        outliers = (data[column] < (q1 - 1.5 * iqr)) | (data[column] > (q3 + 1.5 * iqr))
        count = outliers.sum()
        if count > 0:
            outlier_counts[column] = count

    if outlier_counts:
        print("Outliers detected in the following columns:\n")
        for col, count in outlier_counts.items():
            print(f'{col}: {count} outliers')
    else:
        print("No outliers detected in the dataset.")
        

def chi_square_test(data: pd.DataFrame, columns: list, target: str) -> pd.DataFrame:
    """
    Performs chi-square tests between specified features and a target column in a DataFrame.
    Args:
        data (pd.DataFrame): Input data.
        columns (list): List of feature column names to test.
        target (str): Target column name.
    Returns:
        pd.DataFrame: Results with chi-square statistics, p-values, and significance.
    """
    
    results = []
    
    for col in columns:
        contingency_table = pd.crosstab(data[col], data[target])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        significant = "Yes" if p_value < 0.05 else "No"
        
        results.append({
                'Feature' : col,
                'Chi-Square statistic' : round(chi2, 4),
                'p-value' : round(p_value, 4),
                'Significant (α=0.05)' : significant
            })
        
    return pd.DataFrame(results).sort_values(by='Chi-Square statistic', ascending=False)
        

def t_test(data: pd.DataFrame, features: list, target: str, ) -> pd.DataFrame:
    def t_test(data: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
        """
        Perform independent t-tests for specified features between groups defined by the target column.
        Args:
            data (pd.DataFrame): Input dataframe containing features and target.
            features (list): List of feature column names to test.
            target (str): Target column name to define groups.
        Returns:
            pd.DataFrame: DataFrame with t-statistics, p-values, and significance for each feature.
        """
    
    results = []
    
    for feature in features:
        groups = data.groupby(target)[feature].apply(list)
        t_stat, p_value = ttest_ind(*groups, equal_var=False)
        significant = "Yes" if p_value < 0.05 else "No"
        
        results.append({
            'Feature': feature,
            't-stat': round(t_stat, 4),
            'p-value': round(p_value, 4),
            'Significant (α=0.05)': significant
        })

    return pd.DataFrame(results).sort_values(by='t-stat', ascending=True)


def confidence_interval_t(data: pd.DataFrame, features: list, target: str, confidence=0.95) -> pd.DataFrame:
    """
    Calculates the mean and confidence interval for specified features grouped by a target column.
    Args:
        data (pd.DataFrame): Input dataframe.
        features (list): List of feature column names.
        target (str): Column name to group by.
        confidence (float, optional): Confidence level for the interval. Default is 0.95.
    Returns:
        pd.DataFrame: DataFrame with feature, group, mean, and confidence interval bounds.
    """
    
    results = []
    
    for feature in features:
        groups = data.groupby(target)[feature].apply(list)
        
        for group, val in groups.items():
            mean = pd.Series(val).mean()
            ci_interval = t.interval(
                confidence,
                df=len(val) - 1,
                loc=mean,
                scale=sem(val)
            )
            
            results.append({
                'Feature': feature,
                'Group': group,
                'Mean': mean,
                'CI Lower': ci_interval[0],
                'CI Upper': ci_interval[1]
            })

    return pd.DataFrame(results)


def confidence_interval_z(data: pd.DataFrame, features: list, target: str, confidence=0.95) -> pd.DataFrame:
    """
    Calculates the confidence interval for the mean of specified features grouped by a target column using the Z-distribution.
    Args:
        data (pd.DataFrame): Input dataframe.
        features (list): List of feature column names to analyze.
        target (str): Column name to group by.
        confidence (float, optional): Confidence level for the interval. Default is 0.95.
    Returns:
        pd.DataFrame: DataFrame with feature, group, mean, and confidence interval bounds.
    """

    results = []

    z_score = norm.ppf(1 - (1 - confidence) / 2)

    for feature in features:
        grouped = data.groupby(target)[feature].apply(list)

        for group, values in grouped.items():
            values = pd.Series(values)
            mean = values.mean()
            std = values.std(ddof=1)
            n = len(values)
            margin = z_score * (std / np.sqrt(n))

            results.append({
                'Feature': feature,
                'Group': group,
                'Mean': mean,
                'CI Lower': mean - margin,
                'CI Upper': mean + margin
            })

    return pd.DataFrame(results)


def metrics_over_thresholds(model, X_test, y_test, thresholds):
    """
    Computes classification metrics over a range of probability thresholds.

    Args:
        model: Trained classifier with a predict_proba method.
        X_test: Test feature data.
        y_test: True labels for test data.
        thresholds: Iterable of probability thresholds.

    Returns:
        pd.DataFrame: DataFrame with metrics for each threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        rows.append({
            "threshold": t,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        })
    return pd.DataFrame(rows).sort_values("threshold")


def risk_score_weighted(row, score_weights=risk_score_weighted):
    """
    Calculates a weighted risk score for a given row based on provided score weights.

    Args:
        row: The data row to evaluate.
        score_weights (iterable): An iterable of (name, condition, weight) tuples.

    Returns:
        int or float: The total weighted risk score.
    """
    score = 0
    for name, condition, weight in score_weights:
        if condition(row):
            score += weight
    return score
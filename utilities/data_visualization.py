import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import cross_val_predict


plots_color = "#00BFFF"

def continuous_distribution_plot(data: pd.DataFrame, features: list, hue: str, color=plots_color) -> None:
    """Plots the distribution of continuous features in a DataFrame using histograms with KDE and boxplots.
    Args:
        data (pd.DataFrame): The input DataFrame containing the features to plot.
        features (list): List of column names (strings) representing continuous features to visualize.
        hue (str): The name of the column to use for color encoding.
        color (str, optional): Color to use for the plots. Defaults to plots_color.
    Returns:
        None"""

    n_features = len(features)
    fig, ax = plt.subplots(nrows=n_features, ncols=2, figsize=(15, 5 * n_features))
    
    if n_features == 1:
        ax = [ax]

    for i, feature in enumerate(features):
        avg = data[feature].mean()
        mode = data[feature].mode()[0]
        median = data[feature].median()

        sns.histplot(
            data=data,
            x=feature,
            ax=ax[i][0],
            kde=True,
            color=color,
        )
        
        ax[i][0].axvline(avg, color='red', linestyle="-", label=f"Average: {avg:.2f}", linewidth=2)
        ax[i][0].axvline(mode, color='navy', linestyle="-", label=f"Mode: {mode:.2f}", linewidth=2)
        ax[i][0].axvline(median, color='gold', linestyle="-", label=f"Median: {median:.2f}", linewidth=2)
        ax[i][0].legend()
        
        sns.boxplot(
            data=data,
            x=feature,
            ax=ax[i][1],
            medianprops=dict(color="red", linewidth=2),
            capprops=dict(color="red", linewidth=2),
            color=color,
            hue=hue,
            gap=0.2,
        )

        
    fig.suptitle("Continuous Features Distribution", fontsize=16, fontweight='bold')
        
    sns.despine(top=True, right=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    

def categorical_distribution_plot(data: pd.DataFrame, features: list, hue: str, color=plots_color) -> None:
    """
    Plots the percentage distribution of categorical features in a DataFrame.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing the features to plot.
        features (list): List of column names (strings) representing categorical features to visualize.
        color (str, optional): Color to use for the plots. Defaults to plots_color.
        
    Returns:
        None
    """
    n_features = len(features)
    fig, ax = plt.subplots(nrows=1, ncols=n_features, figsize=(15, 5))
    
    if n_features == 1:
        ax = [ax]

    for i, feature in enumerate(features):
        grouped = (
            data.groupby([feature, hue])
            .size()
            .reset_index(name='count')
        )
        
        total_per_hue = grouped.groupby(hue)['count'].transform('sum')
        grouped['Percentage'] = grouped['count'] / total_per_hue * 100
        
        category_order = (
            grouped.groupby(feature)['Percentage']
            .sum()
            .sort_values(ascending=False)
            .index
        )

        sns.barplot(
            data=grouped,
            x=feature,
            y='Percentage',
            hue=hue,
            ax=ax[i],
            color=color,
            gap=0.2,
            order=category_order
        )

        if i == 1:
            ax[i].legend(title=hue, loc='upper right')
        else:
            ax[i].get_legend().remove()

        for container in ax[i].containers:
            ax[i].bar_label(container, fmt='%.2f%%', padding=5, fontsize=9, rotation=45)

        ax[i].set_ylabel("Percentage")
        ax[i].set_xlabel(None)
        ax[i].tick_params(axis='x', rotation=45)
        ax[i].set_title(f"{feature.replace('_', ' ').capitalize()}", pad=30)

    fig.suptitle("Categorical Features Distribution, %", fontsize=16, fontweight='bold')
    sns.despine(top=True, right=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    

def binary_distribution_plot_percentage(data: pd.DataFrame, features: list, color=plots_color) -> None:
    """
    Plots categorical/binary features 2 per row, one plot each.

    Args:
        data (pd.DataFrame)
        features (list): list of feature names
        color (str): color for plots

    Returns:
        None
    """
    n_features = len(features)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 5 * n_rows))

    if n_rows == 1:
        axes = [ax]

    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]

        counts = data[feature].value_counts(normalize=True).sort_values(ascending=False)
        counts_pct = counts * 100

        sns.barplot(
            x=counts.index, 
            y=counts_pct.values, 
            ax=ax, 
            color=color,
            gap=0.1
        )
        
        ax.set_title(feature.replace('_', ' ').capitalize(), pad=20)
        ax.set_ylabel("Percentage")
        ax.set_xlabel(None)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3)

    fig.suptitle("Binary Features Distribution, %", fontsize=16, fontweight='bold')
    sns.despine(top=True, right=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    
def binary_distribution_plot(data: pd.DataFrame, features: list, hue: str, color=plots_color) -> None:
    """
    Plots categorical/binary features 2 per row, one plot each, grouped by hue.

    Args:
        data (pd.DataFrame): Input data
        features (list): List of binary/categorical feature names
        hue (str): Column name to group bars by
        color (str or list): Color palette

    Returns:
        None
    """
    
    n_features = len(features)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 5 * n_rows))
    axes = ax.flatten() if n_rows > 1 else [ax]

    for i, feature in enumerate(features):
        ax = axes[i]

        sns.countplot(
            data=data,
            x=feature,
            hue=hue,
            ax=ax,
            color=color,
            gap=0.1,
        )

        ax.set_title(feature.replace('_', ' ').capitalize(), pad=20)
        ax.set_ylabel("Count")
        ax.set_xlabel(None)
        ax.grid(axis='y', linestyle='--', alpha=0.2)

        if i == 1:
            ax.legend(title=hue, loc="upper right")
        else:
            ax.get_legend().remove()
            
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3)

    fig.suptitle("Binary Features Distribution by Gender", fontsize=16, fontweight='bold')
    sns.despine(top=True, right=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def feature_correlation_plot(data: pd.DataFrame) -> None:
    """
    Generates and displays a heatmap of the correlation matrix for the given DataFrame.
    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data for which the correlation heatmap is to be generated.
    Returns:
        None: This function does not return any value. It displays the heatmap plot.
        The heatmap uses the 'coolwarm' colormap and annotates the correlation coefficients on the heatmap.
        The plot is titled 'Correlation Heatmap of Red Wine Features' and is formatted with specific font sizes and layout adjustments.
    """
    plt.figure(figsize=(15, 8))
    
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = "Blues"

    sns.heatmap(
        corr, 
        annot=True, 
        mask=mask,
        cmap=cmap, 
        fmt=".3f",
        linewidths=0.5
    )
    
    plt.title("Feature Correlation Heatmap", fontsize=18, fontweight='bold', pad=20)
    
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(models: dict, X_train: pd.DataFrame, y_train: pd.Series, cv: int) -> None:
    """
    Plots confusion matrices with recall scores for multiple classification models using cross-validation.
    Args:
        models (dict): A dictionary where keys are model identifiers and values are tuples of (display_name, model_instance).
        X_train (pd.DataFrame): Feature data for training.
        y_train (pd.Series): Target labels for training.
        cv (int): Number of cross-validation folds.
    Returns:
        None: Displays the confusion matrices as subplots.
    Each subplot shows the confusion matrix and recall score for a model, with axes labeled as 'Predicted' and 'Actual'.
    """
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    
    for m_name, (name, model) in enumerate(models.items()):
        y_pred = cross_val_predict(model, X_train, y_train, cv=cv)
        cm = confusion_matrix(y_train, y_pred)
        
        sns.heatmap(
            cm,
            fmt='d',
            cmap= 'Blues',
            cbar=False,
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'],
            ax=ax[m_name],
            annot=True
        )

        ax[m_name].set_title(f'{name} Confusion Matrix')
        ax[m_name].set_xlabel('Predicted')
        ax[m_name].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    
def final_confusion_matrix(model, y_test, y_pred) -> None:
    """
    Plots the confusion matrix for the final model.

    Args:
        model: The trained model.
        y_test: The true labels for the test set.
        y_pred: The predicted labels for the test set.

    Returns:
        None
    """

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(
        cm,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Stroke', 'Stroke'],
        yticklabels=['No Stroke', 'Stroke'],
        cbar=False,
        annot=True
    )

    ax.set_title(f'Final {model.__class__.__name__} Confusion Matrix\nRecall: {recall:.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.show()
    
    
def plot_confidence_intervals(df: pd.DataFrame) -> None:
    """
    Plots the confidence intervals for the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        None
    """
    features = df["Feature"].unique()
    n = len(features)

    fig, ax = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)

    if n == 1:
        ax = [ax]
        
    for i, f in zip(ax, features):
        sub = df[df["Feature"] == f]
        x = sub["Group"].astype(str)
        y = sub["Mean"]
        yerr = sub["CI Upper"] - sub["Mean"]
        
        i.bar(
            x, 
            y, 
            yerr=yerr,
            capsize=5, 
            color="deepSkyBlue", 
            edgecolor="black"
        )

        
        i.set_title(f'{f.capitalize().replace("_", " ")} Confidence Intervals', fontsize=14, fontweight='bold', pad=20)
        i.set_ylabel('Mean')
        i.set_xlabel('Group')
        i.spines['right'].set_visible(False)
        i.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_permutation_importance(X_test, importances):
    """
    Plots the permutation importance for the given features.

    Args:
        X_test (pd.DataFrame): The test set features.
        importances (np.ndarray): The permutation importances.

    Returns:
        None
    """
    sorted_idx = importances.argsort()

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.barh(
        X_test.columns[sorted_idx], 
        importances[sorted_idx],
        color="deepSkyBlue",
        edgecolor="black"
    )
    
    ax.set_xlabel("Permutation Importance (mean decrease in score)")
    ax.set_title("Feature Importance on Final Model", fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()
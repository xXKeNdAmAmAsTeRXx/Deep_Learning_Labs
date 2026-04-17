import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# Prompt: write in python a Kolmogorov-Smirnov test for distribution normality along with prob plot and hist visualization (function) - Gemini (thinking)
def check_normality(data, label="Dataset"):
    """
    Performs K-S test for normality and visualizes the distribution.
    """
    # 1. Calculate parameters for the comparison distribution
    mu = np.mean(data)
    std = np.std(data, ddof=1)

    # 2. Perform Kolmogorov-Smirnov Test
    # We compare the data to a normal distribution with the sample's mean and std
    statistic, p_value = stats.kstest(data, 'norm', args=(mu, std))

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with KDE
    sns.histplot(data, kde=True, ax=axes[0], color='royalblue', stat="density")
    axes[0].set_title(f'Histogram & KDE: {label}')

    # Probability Plot (Q-Q Plot)
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('royalblue')
    axes[1].set_title(f'Probability Plot: {label}')

    plt.tight_layout()
    plt.show()

    # Output results
    print(f"--- Normality Test Results: {label} ---")
    print(f"K-S Statistic: {statistic:.4f}")
    print(f"P-value:       {p_value:.4f}")

    if p_value > 0.05:
        print("Result: Fail to reject the null hypothesis (Data looks normal).")
    else:
        print("Result: Reject the null hypothesis (Data does NOT look normal).")


# Prompt: write a python function to check correlation of numerical features (Pearson), function should return heatmap along with annotations
def plot_correlation_heatmap(df, title="Correlation Heatmap", size=(12, 10)):
    """
    Calculates Pearson correlation for numerical features and plots a heatmap.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    title (str): The title of the plot.
    size (tuple): The figure size (width, height).
    """
    # 1. Select only numerical columns and calculate Pearson correlation
    corr_matrix = df.select_dtypes(include=['number']).corr(method='pearson')

    # 2. Set up the matplotlib figure
    plt.figure(figsize=size)

    # 3. Create the heatmap
    # annot=True: Adds the numerical values
    # cmap='coolwarm': High contrast for positive/negative correlations
    # fmt=".2f": Rounds annotations to 2 decimal places
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": .8}
    )

    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.show()
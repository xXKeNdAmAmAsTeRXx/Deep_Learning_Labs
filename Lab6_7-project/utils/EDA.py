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


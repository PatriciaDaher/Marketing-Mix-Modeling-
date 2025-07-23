# Marketing Mix Model (MMM) for Digital Channels (Google, Meta, TikTok)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize
import pymc3 as pm
import arviz as az


# Set random seed for reproducibility
np.random.seed(42)

# ======================
# 1. DATA GENERATION (SIMULATED DIGITAL MARKETING DATA)
# ======================
def generate_digital_mmm_data(n_periods=104):
    """Generate synthetic digital marketing data (Google, Meta, TikTok)"""
    dates = pd.date_range(start="2022-01-01", periods=n_periods, freq="W")
    
    # Digital channels (Google, Meta, TikTok)
    data = pd.DataFrame(index=dates)
    
    # Spend data (simulating campaigns)
    for channel in ["Google", "Meta", "TikTok"]:
        # Base spend + seasonality + random spikes
        base = np.random.normal(loc=5000, scale=1000, size=n_periods)
        seasonality = 1000 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        campaigns = np.random.poisson(lam=2000, size=n_periods) * np.random.choice([0, 1], size=n_periods, p=[0.8, 0.2])
        data[f"{channel}_Spend"] = np.clip(base + seasonality + campaigns, 0, 10000)
    
    # Other variables
    data["Clicks"] = np.random.poisson(lam=5000, size=n_periods)  # Proxy for engagement
    data["Competitor_Spend"] = np.random.normal(6000, 1500, size=n_periods)  # Competitor activity
    data["Holiday"] = (data.index.month.isin([11, 12])).astype(int)  # Q4 holidays
    
    # True coefficients (unknown in real scenarios)
    true_coefs = {
        "Google": 0.8,
        "Meta": 1.2,
        "TikTok": 1.5,
        "Clicks": 0.3,
        "Competitor_Spend": -0.5,
        "Holiday": 2.0
    }
    
    # Adstock transformation (digital ads decay faster)
    def adstock(x, theta=0.3, L=3):
        x = np.array(x)
        x_adstock = np.zeros_like(x)
        for t in range(len(x)):
            for l in range(L + 1):
                if t - l >= 0:
                    x_adstock[t] += x[t - l] * (theta ** l)
        return x_adstock
    
    # Saturation (Hill function)
    def hill(x, alpha=2, gamma=3000):
        return (x ** alpha) / (x ** alpha + gamma ** alpha)
    
    # Simulate conversions (target variable)
    conversions = 1000  # Baseline
    
    for channel in ["Google", "Meta", "TikTok"]:
        spend = data[f"{channel}_Spend"]
        adstocked = adstock(spend, theta=np.random.uniform(0.2, 0.4))  # Faster decay for digital
        saturated = hill(adstocked, alpha=2, gamma=5000)
        conversions += true_coefs[channel] * saturated * 100  # Scale for realistic numbers
    
    # Add other effects
    conversions += true_coefs["Clicks"] * data["Clicks"] * 0.1
    conversions += true_coefs["Competitor_Spend"] * (data["Competitor_Spend"] / 1000)
    conversions += true_coefs["Holiday"] * data["Holiday"] * 200
    
    # Add noise
    data["Conversions"] = np.clip(conversions + np.random.normal(0, 200, size=n_periods), 0, None)
    
    return data, true_coefs

data, true_coefs = generate_digital_mmm_data()
print(data.head())

# ======================
# 2. EXPLORATORY ANALYSIS
# ======================
def plot_digital_trends(data):
    """Plot digital spend vs. conversions"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    data["Conversions"].plot(ax=axes[0], title="Conversions Over Time")
    data["Google_Spend"].plot(ax=axes[1], title="Google Spend")
    data["Meta_Spend"].plot(ax=axes[2], title="Meta Spend")
    data["TikTok_Spend"].plot(ax=axes[3], title="TikTok Spend")
    plt.tight_layout()
    plt.show()

plot_digital_trends(data)

# ======================
# 3. FEATURE ENGINEERING
# ======================
def apply_digital_adstock(df, channels, theta=0.3, L=3):
    """Digital ads have faster decay (lower theta)"""
    df = df.copy()
    for channel in channels:
        spend = df[f"{channel}_Spend"]
        adstocked = np.zeros_like(spend)
        for t in range(len(spend)):
            for l in range(L + 1):
                if t - l >= 0:
                    adstocked[t] += spend[t - l] * (theta ** l)
        df[f"{channel}_Adstock"] = adstocked
    return df

def apply_hill_saturation(df, channels, alpha=2, gamma=5000):
    """Apply Hill saturation for digital channels"""
    df = df.copy()
    for channel in channels:
        adstocked = df[f"{channel}_Adstock"]
        df[f"{channel}_Saturated"] = (adstocked ** alpha) / (adstocked ** alpha + gamma ** alpha)
    return df

channels = ["Google", "Meta", "TikTok"]
data = apply_digital_adstock(data, channels)
data = apply_hill_saturation(data, channels)

# ======================
# 4. MODEL TRAINING (LINEAR REGRESSION)
# ======================
features = [f"{c}_Saturated" for c in channels] + ["Clicks", "Competitor_Spend", "Holiday"]
target = "Conversions"

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Train MAPE: {mean_absolute_percentage_error(y_train, train_pred):.2%}")
print(f"Test MAPE: {mean_absolute_percentage_error(y_test, test_pred):.2%}")

# Plot coefficients
coefs = pd.DataFrame({
    "Channel": features,
    "Impact": model.coef_
}).sort_values("Impact", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Impact", y="Channel", data=coefs)
plt.title("Digital Channel Impact on Conversions")
plt.show()

# ======================
# 5. BAYESIAN MMM (PYMC3)
# ======================
def bayesian_digital_mmm(data, channels):
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=1000, sigma=500)
        coefs = {}
        
        for channel in channels:
            coefs[channel] = pm.HalfNormal(f"coef_{channel}", sigma=2)
        
        clicks_coef = pm.HalfNormal("clicks_coef", sigma=1)
        competitor_coef = pm.Normal("competitor_coef", mu=-0.5, sigma=0.2)
        holiday_coef = pm.HalfNormal("holiday_coef", sigma=1)
        
        # Media contributions
        contributions = []
        for channel in channels:
            adstocked = data[f"{channel}_Adstock"].values
            saturated = (adstocked ** 2) / (adstocked ** 2 + 5000 ** 2)
            contributions.append(coefs[channel] * saturated)
        
        # Model formula
        mu = (
            intercept +
            sum(contributions) +
            clicks_coef * (data["Clicks"].values / 100) +
            competitor_coef * (data["Competitor_Spend"].values / 1000) +
            holiday_coef * data["Holiday"].values
        )
        
        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=200)
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=data["Conversions"].values)
        
        # Sampling
        trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.9)
    
    return model, trace

# Run Bayesian model (on first 52 weeks for speed)
model_bayes, trace = bayesian_digital_mmm(data.iloc[:52], channels)

# Plot posterior distributions
az.plot_trace(trace, var_names=["intercept", "coef_Google", "coef_Meta", "coef_TikTok"])
plt.show()

# ======================
# 6. BUDGET OPTIMIZATION
# ======================
def digital_response(spend, coef, theta=0.3, alpha=2, gamma=5000):
    """Calculate response for digital channels"""
    adstocked = spend * (1 - theta ** 4) / (1 - theta)  # Simplified adstock
    saturated = (adstocked ** alpha) / (adstocked ** alpha + gamma ** alpha)
    return coef * saturated * 1000  # Scale to conversions

def optimize_digital_budget(coefs, total_budget=30000):
    """Optimize budget allocation across digital channels"""
    def objective(x):
        total_conversions = 0
        for i, channel in enumerate(channels):
            total_conversions += digital_response(x[i], coefs[i])
        
        # Penalize budget over/under allocation
        penalty = 1000 * abs(sum(x) - total_budget)
        return -(total_conversions - penalty)  # Minimize negative conversions
    
    # Constraints
    bounds = [(1000, 15000) for _ in channels]  # Min $1k, Max $15k per channel
    x0 = [total_budget / len(channels)] * len(channels)  # Start with equal allocation
    
    # Optimize
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda x: sum(x) - total_budget},
        method="SLSQP"
    )
    
    return result.x

# Get coefficients from linear regression
coef_values = [model.coef_[features.index(f"{c}_Saturated")] for c in channels]

# Optimize
optimal_spend = optimize_digital_budget(coef_values)
print("\nOptimal Digital Budget Allocation:")
for channel, spend in zip(channels, optimal_spend):
    print(f"{channel}: ${spend:.2f} ({(spend / 30000):.1%})")

# Plot response curves
spend_range = np.linspace(1000, 15000, 50)
plt.figure(figsize=(10, 6))
for i, channel in enumerate(channels):
    response = [digital_response(x, coef_values[i]) for x in spend_range]
    plt.plot(spend_range, response, label=channel)
plt.title("Digital Channel Response Curves")
plt.xlabel("Spend ($)")
plt.ylabel("Conversions")
plt.legend()
plt.grid()
plt.show()
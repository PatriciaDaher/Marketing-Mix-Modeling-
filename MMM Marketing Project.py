import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
import pymc3 as pm
import arviz as az

# Set random seed for reproducibility
np.random.seed(42)

## 1. Generate Synthetic Data (or load your actual data)
def generate_mmm_data(n_periods=104, channels=['TV', 'Digital', 'Print', 'Radio']):
    """Generate synthetic marketing mix data"""
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='W')
    
    # Generate media spend data with some seasonality
    data = pd.DataFrame(index=dates)
    
    for channel in channels:
        # Base spend with channel-specific patterns
        base = np.random.normal(loc=50, scale=10, size=n_periods)
        
        # Add seasonality
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        
        # Add some spikes for campaigns
        campaigns = np.random.choice([0, 20], size=n_periods, p=[0.9, 0.1])
        
        data[f'{channel}_Spend'] = np.clip(base + seasonality + campaigns, 0, 100)
    
    # Add price and promotion variables
    data['Price'] = np.random.uniform(8, 12, size=n_periods)
    data['Promo'] = np.random.choice([0, 1], size=n_periods, p=[0.7, 0.3])
    
    # Add external factors
    data['Competitor_Activity'] = np.random.normal(0, 1, size=n_periods)
    data['Holiday'] = (data.index.month.isin([11, 12])).astype(int)
    
    # Generate sales (target variable) with some noise
    # True coefficients for each channel (unknown in real scenarios)
    true_coefs = {
        'TV': 0.8,
        'Digital': 1.2,
        'Print': 0.3,
        'Radio': 0.5,
        'Price': -1.5,
        'Promo': 3.0
    }
    
    # Adstock transformation function
    def adstock(x, theta=0.5, L=4):
        x = np.array(x)
        x_adstock = np.zeros_like(x)
        for t in range(len(x)):
            for l in range(L+1):
                if t-l >= 0:
                    x_adstock[t] += x[t-l] * (theta ** l)
        return x_adstock
    
    # Saturation transformation (Hill function)
    def hill(x, alpha=2, gamma=0.5):
        return x**alpha / (x**alpha + gamma**alpha)
    
    # Apply transformations and calculate sales
    sales = 100  # baseline sales
    
    for channel in channels:
        channel_spend = data[f'{channel}_Spend']
        adstocked = adstock(channel_spend, theta=np.random.uniform(0.3, 0.7))
        saturated = hill(adstocked, alpha=np.random.uniform(1.5, 3), gamma=np.random.uniform(20, 40))
        sales += true_coefs[channel] * saturated
    
    # Add other effects
    sales += true_coefs['Price'] * data['Price']
    sales += true_coefs['Promo'] * data['Promo']
    
    # Add seasonality and noise
    sales += 10 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
    sales += np.random.normal(0, 5, size=n_periods)
    
    data['Sales'] = np.clip(sales, 0, None)
    
    return data, true_coefs

# Generate data
data, true_coefs = generate_mmm_data()
print(data.head())

## 2. Exploratory Data Analysis
def plot_time_series(data, cols):
    """Plot time series of selected columns"""
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 3*len(cols)))
    for ax, col in zip(axes, cols):
        data[col].plot(ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

# Plot sales and media spend
plot_time_series(data, ['Sales'] + [f'{c}_Spend' for c in ['TV', 'Digital', 'Print', 'Radio']])

# Decompose sales into trend, seasonality, and residuals
result = seasonal_decompose(data['Sales'], model='additive', period=52)
result.plot()
plt.show()

## 3. Feature Engineering
def apply_adstock(df, channels, theta=0.5, L=4):
    """Apply adstock transformation to media variables"""
    df = df.copy()
    for channel in channels:
        spend = df[f'{channel}_Spend']
        adstocked = np.zeros_like(spend)
        for t in range(len(spend)):
            for l in range(L+1):
                if t-l >= 0:
                    adstocked[t] += spend[t-l] * (theta ** l)
        df[f'{channel}_Adstock'] = adstocked
    return df

def apply_hill(df, channels, alpha=2, gamma=50):
    """Apply saturation transformation using Hill function"""
    df = df.copy()
    for channel in channels:
        adstocked = df[f'{channel}_Adstock']
        df[f'{channel}_Saturated'] = adstocked**alpha / (adstocked**alpha + gamma**alpha)
    return df

channels = ['TV', 'Digital', 'Print', 'Radio']
data = apply_adstock(data, channels)
data = apply_hill(data, channels)

# Add other features
data['Price_Elasticity'] = np.log(data['Price'])
data['Week_of_Year'] = data.index.isocalendar().week
data['Month'] = data.index.month

## 4. Model Building
# Split into train and test
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Prepare features and target
features = [f'{c}_Saturated' for c in channels] + ['Price_Elasticity', 'Promo', 'Holiday', 'Competitor_Activity']
X_train, y_train = train[features], train['Sales']
X_test, y_test = test[features], test['Sales']

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate model
train_pred = lr.predict(X_train)
test_pred = lr.predict(X_test)

print(f"Train MAPE: {mean_absolute_percentage_error(y_train, train_pred):.2%}")
print(f"Test MAPE: {mean_absolute_percentage_error(y_test, test_pred):.2%}")

# Plot coefficients
coefs = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefs)
plt.title('Marketing Mix Model Coefficients')
plt.show()

## 5. Bayesian MMM (More Advanced)
def bayesian_mmm(data, channels):
    """Bayesian Marketing Mix Model using PyMC3"""
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', mu=100, sigma=50)
        coefficients = {}
        
        for channel in channels:
            coefficients[channel] = pm.HalfNormal(f'coef_{channel}', sigma=2)
        
        price_coef = pm.Normal('price_coef', mu=-1, sigma=1)
        promo_coef = pm.HalfNormal('promo_coef', sigma=2)
        
        # Media variables
        contributions = []
        for channel in channels:
            adstock = data[f'{channel}_Adstock'].values
            saturated = adstock**2 / (adstock**2 + 50**2)  # Hill transformation
            contributions.append(coefficients[channel] * saturated)
        
        # Model formula
        mu = (
            intercept +
            sum(contributions) +
            price_coef * data['Price_Elasticity'].values +
            promo_coef * data['Promo'].values
        )
        
        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=10)
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data['Sales'].values)
        
        # Inference
        trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.9)
    
    return model, trace

# Run Bayesian model (on subset of data for speed)
model, trace = bayesian_mmm(train.iloc[:52], channels)  # Using first year for demo

# Plot posterior distributions
az.plot_trace(trace, var_names=['intercept', 'coef_TV', 'coef_Digital', 'coef_Print', 'coef_Radio'])
plt.show()

## 6. Budget Optimization
def response_curve(spend, coef, alpha=2, gamma=50, theta=0.5, L=4):
    """Calculate response for given spend using adstock and hill transformations"""
    # Create time series with single spike
    spends = np.zeros(L+1)
    spends[0] = spend
    
    # Apply adstock
    adstocked = 0
    for l in range(L+1):
        adstocked += spends[l] * (theta ** l)
    
    # Apply hill
    saturated = adstocked**alpha / (adstocked**alpha + gamma**alpha)
    
    return coef * saturated

def objective_function(x, coefficients, total_budget):
    """Objective to maximize (total sales)"""
    total_sales = 0
    for i, channel in enumerate(channels):
        total_sales += response_curve(x[i], coefficients[i])
    
    # Penalize if budget not fully allocated
    penalty = 100 * abs(sum(x) - total_budget)
    return -(total_sales - penalty)  # Negative for minimization

# Get coefficients from linear regression
coef_values = [lr.coef_[features.index(f'{c}_Saturated')] for c in channels]
total_budget = sum([data[f'{c}_Spend'].mean() for c in channels])  # Average historical spend

# Set bounds (min 0, max 2x historical average for each channel)
bounds = [(0, 2*data[f'{c}_Spend'].mean()) for c in channels]

# Initial guess (equal allocation)
x0 = [total_budget/len(channels)] * len(channels)

# Optimize
result = minimize(
    objective_function,
    x0,
    args=(coef_values, total_budget),
    bounds=bounds,
    constraints={'type': 'eq', 'fun': lambda x: sum(x) - total_budget},
    method='SLSQP'
)

# Optimal allocation
optimal_spend = dict(zip(channels, result.x))
print("\nOptimal Media Spend Allocation:")
for channel, spend in optimal_spend.items():
    print(f"{channel}: ${spend:.2f} ({(spend/total_budget):.1%})")

# Plot response curves
plt.figure(figsize=(12, 6))
spend_range = np.linspace(0, 150, 50)
for i, channel in enumerate(channels):
    responses = [response_curve(x, coef_values[i]) for x in spend_range]
    plt.plot(spend_range, responses, label=channel)
plt.title('Media Response Curves')
plt.xlabel('Spend')
plt.ylabel('Incremental Sales')
plt.legend()
plt.grid()
plt.show()
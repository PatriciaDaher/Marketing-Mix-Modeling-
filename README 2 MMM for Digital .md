# Marketing Mix Modeling (MMM) for Digital Channels
## Patricia Daher

## Table of Contents
Project Overview
Key Features
Installation
Usage
Data Requirements
Model Outputs
Customization
Contributing
License

## Project Overview
This repository provides a Python-based Marketing Mix Model (MMM) tailored for digital advertising channels (Google Ads, Meta/Facebook Ads, TikTok Ads). It helps marketers:
Measure ROI of each digital channel
Optimize budget allocation for maximum conversions
Simulate "what-if" scenarios for spend adjustments
Quantify adstock and saturation effects
Built with:
Regression modeling (scikit-learn)
Bayesian inference (PyMC3)
Budget optimization (SciPy)

## Key Features
Feature	Description
Digital-Specific Adstock	Models faster decay in digital ads (θ=0.3)
Hill Saturation	Accounts for diminishing returns at high spend
Bayesian Uncertainty	Quantifies confidence in channel impact
Automated Optimization	Recommends optimal spend per channel
Response Curves	Visualizes channel performance at different spend levels
## Installation
Clone the repository
bash
git clone https://github.com/yourusername/digital-mmm.git  
cd digital-mmm  
Install dependencies
bash
pip install -r requirements.txt  
## Usage
1. Run with Synthetic Data (Demo)
bash
python mmm_digital.py  
2. Use Your Own Data
Format your data as data/your_data.csv (see Data Requirements).
Update the data path in mmm_digital.py:
python
data = pd.read_csv("data/your_data.csv", parse_dates=["Date"])  
3. Key Parameters to Adjust
Parameter	Description	Default
channels	List of digital channels	["Google", "Meta", "TikTok"]
theta	Adstock decay rate (lower = faster decay)	0.3
gamma	Saturation point (lower = faster saturation)	5000
## Data Requirements
Required Columns
Column	Example	Description
Date	2023-01-01	Weekly/daily timestamp
{Channel}_Spend	Google_Spend	Ad spend per channel
Conversions	1500	Target KPI (sales, sign-ups, etc.)
Clicks	4500	Optional engagement metric
Example CSV:

csv
Date,Google_Spend,Meta_Spend,TikTok_Spend,Conversions,Clicks  
2023-01-01,4500,3200,1800,1200,5000  
## Model Outputs
1. Performance Metrics
text
Train MAPE: 8.42%  
Test MAPE: 10.15%  
2. Channel Impact Coefficients
https://i.imgur.com/XYZ1234.png

3. Optimal Budget Allocation
text
Google: $12,456 (41.5%)  
Meta: $9,876 (32.9%)  
TikTok: $7,666 (25.6%)  
4. Response Curves
https://i.imgur.com/ABCD5678.png

## Customization
For Different Industries
python
# Retail (longer adstock)  
theta = 0.5  
gamma = 8000  

# Gaming (faster saturation)  
theta = 0.2  
gamma = 3000  
Adding New Channels
Add column (e.g., LinkedIn_Spend) to your data.
Update channels list:
python
channels = ["Google", "Meta", "TikTok", "LinkedIn"]  
Contributing
Contributions welcome!
Report bugs via Issues.
Suggest features or submit Pull Requests.

# License
General Public License
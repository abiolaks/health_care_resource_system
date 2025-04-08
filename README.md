# HealthCare Resource Optimization System
```
To run the code on Terminal
python -m streamlit run health_v4.py
streamlit run health_v4.py.
```

## 1. Comprehensive Documentation
This section details the design, architecture, and workflow of the  Healthcare Resource Optimization System.

### Overview
The solution is a Streamlit-based dashboard that integrates data processing, analytical modeling, resource optimization, and generative AI. It is designed to facilitate the analysis of healthcare data by:

- Preprocessing and enriching raw healthcare data.

- Clustering regions based on key health risk indicators.

- Predicting mortality rates through regression.

- Optimizing healthcare resource allocation with linear programming.

- Generating AI-based recommendations for policy interventions.
```
Project Structure
bash
Copy
.
├── data/
│   └── nigeria_health_data_36_states.csv    # CSV file with Nigeria health data
├── streamlit_app.py                         # Main Streamlit app code (provided below)
└── README.md                                # Project documentation (this content)
```
```
Dependencies
Streamlit: Creates the interactive web application.

pandas & numpy: For data manipulation and preprocessing.

scikit-learn: Implements clustering (KMeans), regression (LinearRegression), and scaling (StandardScaler).

pulp: Provides a linear programming solver for resource optimization.

OpenAI: Connects to OpenAI’s API to generate AI-powered recommendations.

Plotly & Matplotlib/Seaborn: For interactive and static visualizations.

Install these dependencies using pip:
```
```
bash/cmd/powershell
conda env create -f env.yml
```
### Code Modules and Classes
#### A. HealthDataProcessor
##### Purpose:
- Load raw data from a CSV file, clean missing values, and generate additional metrics.

##### Key Methods:

* load_data(): Reads CSV data, cleans missing numeric/non-numeric values, and computes metrics.

* _clean_data(): Fills missing values using means for numeric data and modes for categorical data.

* _calculate_metrics(): Creates a composite Crisis_Index using weighted health indicators and normalizes it to a 0–100 scale.

* get_processed_data(): Returns the processed DataFrame.

* get_scaled_data(features): Returns standardized features.

#### B. HealthClusterer
##### Purpose:
Use KMeans clustering to classify regions into clusters based on health metrics.

##### Key Methods:

* fit(X): Trains the KMeans model on the provided features.

* get_clusters(): Retrieves cluster labels that are added to the data.

#### C. MortalityPredictor
##### Purpose:
Predict mortality rates based on selected features using a simple linear regression model.

##### Key Methods:

* fit(X, y): Fits the Linear Regression model on training data.

* predict(X): Returns predicted mortality rates.

#### D. ResourceOptimizer
##### Purpose:
Optimize the allocation of key healthcare resources (Beds, Doctors, Vaccines) across regions using linear programming.

##### Key Components:

- Decision Variables: Allocation variables for each region and resource.

- Objective Function: Minimizes the weighted crisis index across regions.

##### Constraints:

- Budget constraint based on sample resource costs.

- Minimum coverage constraints, e.g. a minimum number of beds per population.

##### Key Methods:

- build_model(total_budget): Constructs the optimization problem.

- solve(): Solves the problem using PuLP and stores the solution if optimal.

- get_explanations(): Provides a DataFrame with resource allocation for each region along with contextual crisis index information.

#### E. openai_insights Function
##### Purpose:
Uses OpenAI’s GPT-3.5-turbo to generate actionable healthcare policy recommendations.

##### Usage:
Pass a prompt with state-specific data; receive insights to guide intervention strategies.

#### F. Main Function (Streamlit App)
##### Purpose:
- Provides a multi-tabbed interactive dashboard:

- National Overview: Visual metrics and bar charts showing crisis severity.

-Regional Analysis: 3D clustering visualization and cluster characteristics.

- Risk Forecasting: Scatter plots and alerts for mortality risks.

- Optimization Engine: An interactive slider to set the total budget, run the optimization, and visualize recommendations.

- Custom CSS Styling:
Enhances visual appeal with custom styling for metrics, alerts, and plots.

- Sidebar AI Recommendations:
Enables users to select a state and get AI-generated policy insights.

### Workflow Summary
#### Data Loading & Preprocessing:
The HealthDataProcessor handles CSV loading, missing data treatment, and metric computation (including the crisis index).

- Clustering Analysis:
The HealthClusterer applies KMeans clustering on scaled health features to identify state groupings.

- Mortality Prediction:
The MortalityPredictor provides insights by estimating mortality rates based on relevant features.

- Resource Optimization:
The ResourceOptimizer sets up and solves an optimization problem to suggest how to allocate resources (beds, doctors, vaccines) across regions under budget and minimum coverage constraints.

- AI-Powered Recommendations:
The openai_insights function makes a call to OpenAI’s GPT-3.5-turbo API with a relevant prompt to generate tailored recommendations.

- Interactive Visualization:
Streamlit provides tabs, sliders, charts (via Plotly and Matplotlib), and tables to communicate data insights and model outputs.


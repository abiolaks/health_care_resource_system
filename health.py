import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pulp
from openai import OpenAI 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tomllib  # built-in library for Python 3.11+

# Configuration
#openai.api_key = st.secrets["OPENAI_KEY"]



# Set the API key for OpenAI
#openai.api_key = 
client = OpenAI(api_key=st.secrets["openai"]["api_key"])



class HealthDataProcessor:
    """Class to load and preprocess health data"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self._clean_data()

    def _clean_data(self):
        # Handle numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # Handle non-numeric columns
        non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

    def get_processed_data(self):
        return self.data.copy()

    def get_scaled_data(self, features):
        return self.scaler.fit_transform(self.data[features])


class HealthClusterer:
    """Class to perform state clustering"""

    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X):
        self.model.fit(X)

    def get_clusters(self):
        return self.model.labels_


class MortalityPredictor:
    """Class to predict mortality rates"""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ResourceOptimizer:
    """Class to optimize resource allocation"""

    def __init__(self, data):
        self.data = data
        self.problem = pulp.LpProblem("Healthcare_Resource_Allocation", pulp.LpMinimize)

    def build_model(self):
        # Define variables and constraints
        pass  # Implementation details below


def openai_insights(prompt):
    """Get AI-generated insights using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to generate insights: {str(e)}")
        return "Error generating recommendations"


# Streamlit App
def main():
    st.set_page_config(page_title="Nigeria Health Analytics", layout="wide")

    # Load data
    processor = HealthDataProcessor("./data/nigeria_health_data_36_states.csv")
    processor.load_data()
    data = processor.get_processed_data()

    st.title("Nigeria Healthcare Resource Optimization")

    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Clustering", "Predictions", "Optimization"]
    )

    with tab1:
        st.header("National Health Overview")
        fig = px.choropleth(
            data,
            locations="Region Name/Code",
            color="Mortality Rate (per 1000)",
            scope="africa",
            title="Mortality Rate by State",
        )
        st.plotly_chart(fig)

    with tab2:
        st.header("Regional Clustering Analysis")
        features = [
            "Mortality Rate (per 1000)",
            "Healthcare Facilities",
            "Poverty Rate (%)",
        ]
        X = processor.get_scaled_data(features)

        clusterer = HealthClusterer(n_clusters=3)
        clusterer.fit(X)
        data["Cluster"] = clusterer.get_clusters()

        fig = px.scatter(
            data,
            x="Poverty Rate (%)",
            y="Mortality Rate (per 1000)",
            color="Cluster",
            hover_name="Region Name/Code",
        )
        st.plotly_chart(fig)

    with tab3:
        st.header("Mortality Rate Prediction")
        features = [
            "Healthcare Providers",
            "Prevalence - Malaria (%)",
            "Avg Income (NGN/month)",
        ]
        X = data[features]
        y = data["Mortality Rate (per 1000)"]

        predictor = MortalityPredictor()
        predictor.fit(X, y)
        predictions = predictor.predict(X)

        fig, ax = plt.subplots()
        sns.regplot(x=y, y=predictions, ax=ax)
        ax.set_xlabel("Actual Mortality Rates")
        ax.set_ylabel("Predicted Mortality Rates")
        st.pyplot(fig)

    with tab4:
        st.header("Resource Optimization Model")
        # Optimization implementation
        pass

    # OpenAI Insights Section
    st.sidebar.header("AI-Powered Insights")
    selected_state = st.sidebar.selectbox("Select State", data["Region Name/Code"])
    state_data = data[data["Region Name/Code"] == selected_state].iloc[0]

    prompt = f"""Analyze this healthcare data for {selected_state}:
    - Mortality Rate: {state_data['Mortality Rate (per 1000)']}
    - Poverty Rate: {state_data['Poverty Rate (%)']}%
    - Malaria Prevalence: {state_data['Prevalence - Malaria (%)']}%
    - Healthcare Facilities: {state_data['Healthcare Facilities']}
    Provide 3 actionable recommendations to improve healthcare outcomes."""

    if st.sidebar.button("Generate Recommendations"):
        insights = openai_insights(prompt)
        st.sidebar.markdown(f"**Recommendations for {selected_state}:**")
        st.sidebar.write(insights)


if __name__ == "__main__":
    main()

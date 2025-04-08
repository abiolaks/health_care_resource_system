import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pulp
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

# Configuration
#client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
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
        # Handle missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
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
    
    def build_model(self, budget=5e9):
        # Implementation of optimization logic
        pass

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
        return f"Error generating insights: {str(e)}"

def main():
    st.set_page_config(page_title="Nigeria Health Analytics", layout="wide")
    
    """st.markdown(
        """
        #<style>
       # .highlight {
           # background-color: #f0f0f0;
          #  padding: 10px;
           # border-radius: 5px;
           # margin-bottom: 20px;
      #  }
       # </style>
      #  """,
      #  unsafe_allow_html=True,
   # )
        # Custom CSS
    st.markdown("""
    <style>
        .stMetric { border-left: 5px solid #4CAF50; padding: 15px; }
        .stAlert { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stPlotlyChart { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
        .highlight { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)
        
    # Load data
    processor = HealthDataProcessor("./data/nigeria_health_data_36_states.csv")
    processor.load_data()
    data = processor.get_processed_data()

    st.title("🇳🇬 Nigeria Healthcare Resource Optimization System")
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(
        ["National Overview", "Regional Analysis", "Risk Forecasting", "Optimization Engine"]
    )

    with tab1:
        st.header("National Health Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_mortality = data['Mortality Rate (per 1000)'].mean()
            st.metric("Avg Mortality Rate", f"{avg_mortality:.1f}/1000", "National Baseline")
        
        with col2:
            urgent_states = data[data['Mortality Rate (per 1000)'] > 15]['Region Name/Code'].nunique()
            st.metric("Critical States", urgent_states, "High Mortality Regions")
        
        with col3:
            staff_gap = (data['Healthcare Providers'].sum() / 1e3)
            st.metric("Staff Shortage", f"{staff_gap:.1f}K", "Healthcare Providers Needed")
        
        with col4:
            vaccine_cov = data['Vaccination Uptake (%)'].mean()
            st.metric("Vaccination Coverage", f"{vaccine_cov:.1f}%", "National Average")

        # Interactive Map
        fig = px.choropleth(data,
                          locations="Region Name/Code",
                          color="Mortality Rate (per 1000)",
                          hover_name="Region Name/Code",
                          hover_data=["Healthcare Facilities", "Poverty Rate (%)"],
                          scope="africa",
                          color_continuous_scale="RdYlGn_r",
                          title="<b>Healthcare Crisis Severity Map</b>")
        fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Regional Vulnerability Analysis")
        
        # 3D Clustering
        features_3d = ['Mortality Rate (per 1000)', 'Poverty Rate (%)', 'Healthcare Facilities']
        X_3d = processor.get_scaled_data(features_3d)
        
        clusterer = HealthClusterer(n_clusters=4)
        clusterer.fit(X_3d)
        data['Cluster'] = clusterer.get_clusters()
        
        fig = px.scatter_3d(data,
                           x='Mortality Rate (per 1000)',
                           y='Poverty Rate (%)',
                           z='Healthcare Facilities',
                           color='Cluster',
                           hover_name='Region Name/Code',
                           labels={'color': 'Risk Level'},
                           title="3D Regional Vulnerability Clustering")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Insights
        st.subheader("Cluster Characteristics")
        cluster_stats = data.groupby('Cluster').agg({
            'Mortality Rate (per 1000)': 'mean',
            'Poverty Rate (%)': 'mean',
            'Healthcare Facilities': 'median'
        }).reset_index()
        
        st.dataframe(cluster_stats.style
                    .background_gradient(cmap='YlOrBr')
                    .format({"Mortality Rate (per 1000)": "{:.1f}",
                            "Poverty Rate (%)": "{:.1f}%"}),
                    height=300)

    with tab3:
        st.header("Mortality Risk Forecasting")
        
        # Model Explanation
        st.markdown("""
        <div class="highlight">
        <b>Prediction Model Insights:</b><br>
        This model identifies mortality risk drivers using:
        - 🦟 Malaria Prevalence Impact
        - 🏥 Healthcare Staffing Levels
        - 💰 Economic Capacity (Average Income)
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Analysis
        fig = px.scatter(data,
                        x='Prevalence - Malaria (%)',
                        y='Mortality Rate (per 1000)',
                        size='Healthcare Providers',
                        color='Avg Income (NGN/month)',
                        hover_name='Region Name/Code',
                        trendline="ols",
                        title="Mortality Risk Drivers Analysis")
        st.plotly_chart(fig)
        
        # High Risk Alerts
        high_risk = data[data['Mortality Rate (per 1000)'] > data['Mortality Rate (per 1000)'].quantile(0.75)]
        if not high_risk.empty:
            st.warning(f"🚨 Critical Alert: {len(high_risk)} states in top 25% mortality risk")
            st.dataframe(high_risk[['Region Name/Code', 'Mortality Rate (per 1000)', 'Poverty Rate (%)']]
                        .sort_values('Mortality Rate (per 1000)', ascending=False),
                        height=200)

    with tab4:
        st.header("Resource Optimization Engine")
        st.info("🚧 Optimization module under active development - Coming Soon!")
        
    # AI Insights Sidebar
    st.sidebar.header("AI-Powered Recommendations")
    selected_state = st.sidebar.selectbox("Select State", data['Region Name/Code'])
    state_data = data[data['Region Name/Code'] == selected_state].iloc[0]
    
    prompt = f"""As a healthcare policy expert, analyze this data for {selected_state}:
    - Population: {state_data['Population Size']}
    - Mortality Rate: {state_data['Mortality Rate (per 1000)']}/1000
    - Poverty Rate: {state_data['Poverty Rate (%)']}%
    - Healthcare Facilities: {state_data['Healthcare Facilities']}
    - Malaria Prevalence: {state_data['Prevalence - Malaria (%)']}%
    Provide 3 actionable recommendations to improve healthcare outcomes."""
    
    if st.sidebar.button("Generate Recommendations"):
        with st.sidebar:
            with st.spinner("Analyzing with AI..."):
                insights = openai_insights(prompt)
                st.markdown(f"**Recommended Actions for {selected_state}:**")
                st.write(insights)

if __name__ == "__main__":
    main()
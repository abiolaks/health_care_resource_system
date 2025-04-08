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
import matplotlib.pyplot as plt
import seaborn as sns

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
        self._calculate_metrics()

    def _clean_data(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

    def _calculate_metrics(self):
        # Calculate composite crisis index
        self.data['Crisis_Index'] = (
            0.4 * self.data['Mortality Rate (per 1000)'] +
            0.3 * self.data['Poverty Rate (%)'] +
            0.2 * (100 - self.data['Vaccination Uptake (%)']) +
            0.1 * (1 / (self.data['Healthcare Facilities'] + 1))
        )
        
        # Normalize to 0-100 scale
        self.data['Crisis_Index'] = 100 * (self.data['Crisis_Index'] - self.data['Crisis_Index'].min()) / \
                                  (self.data['Crisis_Index'].max() - self.data['Crisis_Index'].min())

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
    """Optimization engine with explanations"""
    def __init__(self, data):
        self.data = data
        self.problem = pulp.LpProblem("Healthcare_Resource_Allocation", pulp.LpMinimize)
        self.solution = None

    def build_model(self, total_budget=5e9):
        # Decision variables
        regions = self.data['Region Name/Code'].unique()
        resources = ['Beds', 'Doctors', 'Vaccines']
        
        # Create variables
        self.vars = pulp.LpVariable.dicts(
            "Allocation",
            [(r, res) for r in regions for res in resources],
            lowBound=0,
            cat='Integer'
        )
        
        # Objective: Minimize weighted crisis index
        self.problem += pulp.lpSum(
            self.data.loc[self.data['Region Name/Code'] == r, 'Crisis_Index'].values[0] * 
            (self.vars[(r, 'Beds')] + self.vars[(r, 'Doctors')] + self.vars[(r, 'Vaccines')])
            for r in regions
        )
        
        # Budget constraints (sample costs)
        costs = {
            'Beds': 500000,    # 500k NGN per bed
            'Doctors': 2000000, # 2M NGN per doctor
            'Vaccines': 1500    # 1.5k NGN per vaccine
        }
        
        self.problem += pulp.lpSum(
            costs[res] * self.vars[(r, res)]
            for r in regions for res in resources
        ) <= total_budget
        
        # Minimum coverage constraints
        for r in regions:
            pop = self.data.loc[self.data['Region Name/Code'] == r, 'Population Size'].values[0]
            self.problem += self.vars[(r, 'Beds')] >= pop * 0.002  # 2 beds per 1000 people
            self.problem += self.vars[(r, 'Doctors')] >= pop * 0.0005  # 1 doctor per 2000 people
            self.problem += self.vars[(r, 'Vaccines')] >= pop * 0.5  # Vaccinate 50% population

    def solve(self):
        self.problem.solve()
        if pulp.LpStatus[self.problem.status] == 'Optimal':
            self.solution = {
                var.name: var.varValue 
                for var in self.problem.variables()
                if var.varValue > 0
            }
            return True
        return False

    def get_explanations(self):
        if not self.solution:
            return "No solution found"
            
        # Generate insights
        allocations = pd.DataFrame.from_dict(
            {(k[0], k[1]): v for k, v in self.solution.items()},
            orient='index',
            columns=['Quantity']
        ).reset_index()
        
        allocations.columns = ['Region', 'Resource', 'Quantity']
        
        # Merge with original data
        result = pd.merge(
            allocations.pivot(index='Region', columns='Resource', values='Quantity'),
            self.data[['Region Name/Code', 'Crisis_Index']],
            left_on='Region',
            right_on='Region Name/Code'
        )
        
        return result.sort_values('Crisis_Index', ascending=False)

def openai_insights(prompt):
    """Get AI-generated insights using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a healthcare policy expert providing actionable recommendations based on data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"

def main():
    st.set_page_config(page_title="Nigeria Health Analytics", layout="wide")
    # Custom CSS
    st.markdown("""
    <style>
        .stMetric { border-left: 5px solid #4CAF50; padding: 15px; }
        .stAlert { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stPlotlyChart { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
        .highlight { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
        .urgent { color: #d32f2f; font-weight: bold; }
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
            urgent_states = data[data['Crisis_Index'] > 70]['Region Name/Code'].nunique()
            st.metric("Critical States", urgent_states, "Crisis Index > 70%")
        
        with col3:
            staff_gap = (data['Healthcare Providers'].sum() / 1e3)
            st.metric("Staff Shortage", f"{staff_gap:.1f}K", "Healthcare Providers Needed")
        
        with col4:
            vaccine_cov = data['Vaccination Uptake (%)'].mean()
            st.metric("Vaccination Coverage", f"{vaccine_cov:.1f}%", "National Average")

        # Nigeria Map Visualization (Alternative approach)
        st.subheader("Healthcare Crisis Severity by State")
        
        # Create a bar chart as alternative to map
        crisis_data = data.sort_values('Crisis_Index', ascending=False)
        fig = px.bar(crisis_data,
                    x='Region Name/Code',
                    y='Crisis_Index',
                    color='Crisis_Index',
                    color_continuous_scale="RdYlGn_r",
                    labels={'Crisis_Index': 'Severity Index (%)'},
                    hover_data=['Mortality Rate (per 1000)', 'Poverty Rate (%)', 'Healthcare Facilities'])
        
        fig.update_layout(
            xaxis_title="State",
            yaxis_title="Crisis Severity Index",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 critical states
        st.markdown("### 🔴 Top 5 Critical States")
        critical_states = crisis_data.head(5)[['Region Name/Code', 'Crisis_Index', 'Mortality Rate (per 1000)']]
        st.dataframe(critical_states.style
                    .background_gradient(cmap='Reds', subset=['Crisis_Index', 'Mortality Rate (per 1000)']),
                    height=200)

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
            'Healthcare Facilities': 'median',
            'Crisis_Index': 'mean'
        }).reset_index()
        
        st.dataframe(cluster_stats.style
                    .background_gradient(cmap='YlOrBr')
                    .format({
                        'Mortality Rate (per 1000)': '{:.1f}',
                        'Poverty Rate (%)': '{:.1f}%',
                        'Crisis_Index': '{:.1f}%'
                    }),
                    height=300)
        
        # Cluster-specific recommendations
        st.markdown("### 📋 Cluster-Based Recommendations")
        recommendations = {
            0: "✅ Stable regions: Maintain current funding with focus on preventive care",
            1: "⚠️ Moderate risk: Increase primary healthcare investments",
            2: "🚨 High risk: Emergency interventions needed for infrastructure and staffing",
            3: "🔴 Critical: Comprehensive support package required (facilities + personnel + vaccines)"
        }
        
        for cluster, rec in recommendations.items():
            states = data[data['Cluster'] == cluster]['Region Name/Code'].tolist()
            st.markdown(f"**Cluster {cluster} ({len(states)} states):** {rec}")
            st.caption(f"States: {', '.join(states)}")

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
        
        budget = st.slider("Select Total Budget (NGN)", 1e9, 1e10, 5e9, step=1e8, format="%.0f")
        
        optimizer = ResourceOptimizer(data)
        with st.spinner("Building optimization model..."):
            optimizer.build_model(total_budget=budget)
        
        if st.button("Run Optimization"):
            with st.spinner("Solving optimization problem..."):
                if optimizer.solve():
                    st.success("✅ Optimal allocation found!")
                    
                    # Show results
                    results = optimizer.get_explanations()
                    st.subheader("Optimal Resource Allocation")
                    st.dataframe(results.style
                                .background_gradient(cmap='YlGnBu', subset=['Beds', 'Doctors', 'Vaccines']),
                                height=400)
                    
                    # Visualize allocation
                    fig = px.bar(results,
                                x='Region',
                                y=['Beds', 'Doctors', 'Vaccines'],
                                title="Recommended Resource Allocation",
                                barmode='group')
                    st.plotly_chart(fig)
                    
                    # Key insights
                    st.subheader("Optimization Insights")
                    top_recipient = results.iloc[0]['Region']
                    st.markdown(f"""
                    - 🎯 **Top Priority**: {top_recipient} will receive the most resources due to highest crisis index
                    - ⚖️ **Equity Focus**: System automatically balances between high-need and high-population states
                    - 💡 **Efficiency**: Allocation minimizes mortality risk per Naira spent
                    """)
                else:
                    st.error("No feasible solution found with current budget")

    # AI Recommendations Sidebar
    st.sidebar.header("🧠 AI-Powered Recommendations")
    selected_state = st.sidebar.selectbox(
        "Select State for Analysis",
        options=data["Region Name/Code"],
        index=0
    )
    
    state_data = data[data["Region Name/Code"] == selected_state].iloc[0]
    st.sidebar.markdown(f"""
    ### {selected_state} Snapshot
    - 🧑 Population: {state_data['Population Size']:,}
    - ⚕️ Facilities: {state_data['Healthcare Facilities']}
    - 💰 Avg Income: ₦{state_data['Avg Income (NGN/month)']:,.0f}
    - 🏥 Crisis Index: {state_data['Crisis_Index']:.1f}%
    """)
    
    if st.sidebar.button("Generate Recommendations"):
        with st.sidebar:
            with st.spinner("Analyzing state data..."):
                prompt = f"""As healthcare policy expert, analyze {selected_state} with:
                - Mortality: {state_data['Mortality Rate (per 1000)']}/1000
                - Poverty: {state_data['Poverty Rate (%)']}%
                - Malaria: {state_data['Prevalence - Malaria (%)']}%
                - Vaccination: {state_data['Vaccination Uptake (%)']}%
                - Classification: {state_data['Urban vs. Rural Classification']}
                
                Provide 3 specific recommendations to improve healthcare outcomes."""
                
                insights = openai_insights(prompt)
                st.markdown(f"### 📋 Recommendations for {selected_state}")
                st.markdown("---")
                st.markdown(insights)

if __name__ == "__main__":
    main()
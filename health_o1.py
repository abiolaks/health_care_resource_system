# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Machine learning models
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

# For generative AI recommendations via OpenAI API (if you have an API key)
from openai import OpenAI 
import os

# -----------------------
# 1. Load and Preprocess Data
# -----------------------

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    data = pd.read_csv("./data/nigeria_health_data_36_states_o1_v2.csv")
    # Make sure column names are consistent (if needed, rename them)
    # For this example, we assume the columns are named as below.
    # Create a Total Resources metric from the available resources:
    data["Total_Resources"] = (data["Medical_Facilities"] +
                               data["Healthcare_Professionals"] +
                               data["Medications"] +
                               data["Emergency_Services"])
    
    # Define an "ideal" resource amount (here, proportional to population using average resource per capita)
    avg_resource_per_capita = data["Total_Resources"].sum() / data["Population"].sum()
    data["Ideal_Resources"] = data["Population"] * avg_resource_per_capita

    # Compute a resource gap (actual - ideal); negative gap indicates under-served regions.
    data["Resource_Gap"] = data["Total_Resources"] - data["Ideal_Resources"]

    # Define a service status based on resource gap
    data["Service_Status"] = data["Resource_Gap"].apply(lambda x: "Under-served" if x < 0 else "Adequately served")
    
    return data

data = load_data()

# -----------------------
# 2. Streamlit App Layout and Sidebar
# -----------------------
st.title("Nigeria Health Resources Analysis & Recommendations")
st.markdown("""
This interactive dashboard analyzes regional health data across Nigeria’s 36 states.
You can explore the data, view model outputs and visualizations, and see recommendations generated for intervention strategies.
""")

# Sidebar options to choose the analysis section
analysis_mode = st.sidebar.radio("Select Analysis Mode:", 
                                 ("Exploratory Data Analysis", "Clustering Analysis", 
                                  "Regression Analysis", "Classification Analysis", "AI Recommendations"))


# -----------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------
if analysis_mode == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(data)

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Resource Distribution Across States")
    fig_bar = px.bar(data, x="State", y="Total_Resources",
                     title="Total Health Resources by State")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Distribution of Service Status")
    fig_pie = px.pie(data, names="Service_Status", title="Service Status Proportion")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = data[["Population", "Medical_Facilities", "Healthcare_Professionals", "Medications", "Emergency_Services", "Total_Resources", "Resource_Gap"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------
# 4. Clustering Analysis (K-Means)
# -----------------------
elif analysis_mode == "Clustering Analysis":
    st.header("Clustering Analysis")
    st.markdown("Using K-Means clustering to group states based on health resources and population.")

    # We choose the features: Population, Total_Resources, Resource_Gap
    features = data[["Population", "Total_Resources", "Resource_Gap"]]
    # Optionally, scale the features (not included here for brevity – you could use StandardScaler)

    # Define number of clusters (e.g., 3 clusters)
    k = st.sidebar.slider("Select number of clusters (K)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    data["Cluster"] = kmeans.fit_predict(features)
    
    st.write("Cluster centroids:")
    centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=["Population", "Total_Resources", "Resource_Gap"])
    st.dataframe(centroid_df)
    
    # Visualize clusters
    fig_cluster = px.scatter_3d(data, x="Population", y="Total_Resources", z="Resource_Gap",
                                color="Cluster", hover_data=["State"],
                                title="3D Scatter Plot of Clusters")
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("States grouped by their cluster assignment:")
    st.dataframe(data[["State", "Cluster"]].sort_values(by="Cluster"))


# -----------------------
# 5. Regression Analysis
# -----------------------
elif analysis_mode == "Regression Analysis":
    st.header("Regression Analysis")
    st.markdown("""
    We use a Random Forest Regression model to predict the resource gap for each state based on its population and available resources.
    This model helps estimate if a state is likely to be under-served or over-served.
    """)
    # Define features and target
    features = data[["Population", "Total_Resources"]]
    target = data["Resource_Gap"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    model_reg.fit(X_train, y_train)
    predictions = model_reg.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"Mean Squared Error on test data: {mse:.2f}")

    # Display a scatter plot for actual vs predicted resource gap
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    fig_reg = px.scatter(df_pred, x="Actual", y="Predicted", 
                         title="Actual vs Predicted Resource Gap",
                         trendline="ols")
    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("This model helps identify potential resource deficiencies that need attention.")


# -----------------------
# 6. Classification Analysis
# -----------------------
elif analysis_mode == "Classification Analysis":
    st.header("Classification Analysis")
    st.markdown("""
    Here we build two classification models to predict the service status (whether a state is "Under-served" or "Adequately served").
    We first compute the service status based on the resource gap.
    """)
    # Prepare feature set and target
    clf_features = data[["Population", "Total_Resources"]]
    # Our target is already computed as "Service_Status"
    clf_target = data["Service_Status"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(clf_features, clf_target, test_size=0.2, random_state=42)

    # Model 1: Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_predictions = dt_clf.predict(X_test)
    st.subheader("Decision Tree Classifier Results")
    st.text(classification_report(y_test, dt_predictions))
    cm_dt = confusion_matrix(y_test, dt_predictions)
    fig_cm_dt = px.imshow(cm_dt, text_auto=True, 
                          labels=dict(x="Predicted", y="Actual"),
                          x=["Adequately served", "Under-served"],
                          y=["Adequately served", "Under-served"],
                          title="Confusion Matrix - Decision Tree")
    st.plotly_chart(fig_cm_dt, use_container_width=True)

    # Model 2: Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(random_state=42)
    gb_clf.fit(X_train, y_train)
    gb_predictions = gb_clf.predict(X_test)
    st.subheader("Gradient Boosting Classifier Results")
    st.text(classification_report(y_test, gb_predictions))
    cm_gb = confusion_matrix(y_test, gb_predictions)
    fig_cm_gb = px.imshow(cm_gb, text_auto=True, 
                          labels=dict(x="Predicted", y="Actual"),
                          x=["Adequately served", "Under-served"],
                          y=["Adequately served", "Under-served"],
                          title="Confusion Matrix - Gradient Boosting")
    st.plotly_chart(fig_cm_gb, use_container_width=True)

    st.markdown("These models help identify which states are likely under-served and could benefit from additional resources.")


# -----------------------
# 7. AI Recommendations via Gen AI
# -----------------------
elif analysis_mode == "AI Recommendations":
    st.header("Generative AI Recommendations")
    st.markdown("""
    Based on the data analysis and model outputs, here are some tailored recommendations on resource reallocation and intervention strategies.
    The recommendations are generated using a GPT model and are meant to guide your decision-making.
    """)
    
    # Prompt generation based on our findings (example text).
    recommendation_prompt = f"""
    Based on the provided Nigeria health dataset across 36 states, the analysis shows the following:
    
    - Some states are under-served, with a negative resource gap.
    - Clustering has grouped states based on population and total resources.
    - Regression analysis predicts resource gaps accurately.
    - Classification models indicate which states are under-served.
    
    Please provide detailed recommendations for ensuring efficient allocation of:
    - Medical facilities
    - Healthcare professionals
    - Medications
    - Emergency services
    
    Include potential intervention strategies that address the specific needs of under-served regions.
    """

    # Check if the user provided an OpenAI API key using secrets (or enter manually)
    client = OpenAI(api_key=st.secrets["openai"]["api_key"]) 
    
    if client:
        st.subheader("Generated Recommendations")
        st.text_area("AI Recommendations Prompt", value=recommendation_prompt, height=200)
        try:
            response = client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a healthcare domain expert providing recommendations for resource allocation."},
                    {"role": "user", "content": recommendation_prompt}
                ]
            )
            recommendation_text = response.choices[0].message.content.strip()
            st.subheader("Generated Recommendations")
            st.write(recommendation_text)
        except Exception as e:
            st.error(f"Error calling the OpenAI API: {e}")
            st.info("Please check your API key or network connection.")
    else:
        st.info("Enter an OpenAI API key above to generate AI recommendations.")
        st.markdown("""
        **Example Recommendations Preview:**

        - **Reallocation of Resources:** Increase the number of mobile clinics and telemedicine services in states identified as under-served.
        - **Workforce Development:** Invest in local healthcare training programs and provide incentives for professionals to work in high-need areas.
        - **Supply Chain Interventions:** Improve logistics and distribution networks for essential medications, especially in rural regions.
        - **Emergency Preparedness:** Enhance emergency services by establishing regional emergency hubs and improving on-call medical staffing.
        """)

# -----------------------
# End of Streamlit App
# -----------------------

st.sidebar.markdown("### Additional Information")
st.sidebar.info("This interactive dashboard uses multiple analytical models and generative AI to provide recommendations based on Nigeria's health resource data. Adjust the inputs via the sidebar to explore different analyses.")

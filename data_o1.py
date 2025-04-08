import pandas as pd
import numpy as np

# List of Nigeria's 36 states.
states = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos",
    "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers",
    "Sokoto", "Taraba", "Yobe", "Zamfara"
]

# Set random seed for reproducibility.
np.random.seed(42)

# Prepare an empty list to hold the data rows.
data_rows = []

for state in states:
    # ------------------------------
    # 1. Population: Generate a random population number.
    #    We assume states have populations between 1,000,000 and 20,000,000.
    # ------------------------------
    population = np.random.randint(1_000_000, 20_000_000)
    
    # ------------------------------
    # 2. Medical Facilities: Assume on average, there is 1 facility per 50,000 inhabitants.
    #    We then add some random variation.
    # ------------------------------
    base_facilities = population / 50000  # Ideal count based on population.
    # Multiply by a random normal factor to simulate differences in resource allocation.
    facilities = int(np.round(base_facilities * np.random.normal(1, 0.2)))
    facilities = max(1, facilities)  # Ensure at least one facility.
    
    # ------------------------------
    # 3. Healthcare Professionals: Assume between 8 to 15 professionals per facility.
    # ------------------------------
    hp_per_facility = np.random.randint(8, 16)
    healthcare_professionals = facilities * hp_per_facility
    
    # ------------------------------
    # 4. Medications: Assume that each facility is associated with roughly 100 units of medication supply,
    #    with some random variation of ±10%.
    # ------------------------------
    med_multiplier = np.random.uniform(0.9, 1.1)
    medications = int(np.round(facilities * 100 * med_multiplier))
    
    # ------------------------------
    # 5. Emergency Services: Assume roughly one emergency service unit per 250,000 inhabitants,
    #    also varied by ±20% to simulate differences in emergency preparedness.
    # ------------------------------
    base_emergency = population / 250000
    emergency_factor = np.random.normal(1, 0.2)
    emergency_services = int(np.round(base_emergency * emergency_factor))
    emergency_services = max(1, emergency_services)  # Ensure at least one service unit.
    
    # Append the generated row as a dictionary.
    data_rows.append({
        "State": state,
        "Population": population,
        "Medical_Facilities": facilities,
        "Healthcare_Professionals": healthcare_professionals,
        "Medications": medications,
        "Emergency_Services": emergency_services
    })

# Convert the list of rows into a pandas DataFrame.
df = pd.DataFrame(data_rows)

# Display the first few rows of the generated DataFrame.
print(df.head())

# Save the DataFrame to a CSV file.
df.to_csv("./data/nigeria_health_data_36_states_o1_v2.csv", index=False)# from o1 model

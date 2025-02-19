import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title('Parking Structure Cost of Construction Analytics')

# Define initial weights (total = 100)
initial_weights = {
    'Design Efficiency': 12,            # Major impact on cost
    'Type of Construction': 8,          # Significant structural impact
    'Non-parking Areas': 4,
    'Double-loaded Bays': 4,
    'Parking Ramps': 4,
    'Parking Controls': 2,
    'Parking Footprint': 4,
    'Subterranean Levels': 8,          # Major cost factor
    'Design Redundancy': 4,
    'Soil Condition': 6,               # Important for foundation costs
    'Location Cost': 8,                # Major regional cost factor
    'Seismic Zone': 6,                # Important for structural requirements
    'Site Access': 2,
    'Number of Bidders': 2,
    'Economic Factors': 4,
    'Type of Delivery': 2,
    'Façade Treatment': 4,
    'Ventilation Type': 2,
    'Sprinkler System': 2,
    'Utilities Inclusion': 2,
    'Construction Method': 4,          # Important for timeline and cost
    'Structural System': 4,            # Important for seismic and durability
    'Time Cost Escalation': 2         # Market timing impact
}

# Verify that weights sum to 100
assert sum(initial_weights.values()) == 100, "Initial weights must sum to 100"

# Initialize session state for weights if not already set
for key, value in initial_weights.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to adjust weights
def adjust_weights(key, new_value):
    # Calculate the total weight excluding the current key
    total_excluding_current = sum(st.session_state[k] for k in initial_weights if k != key)
    # Calculate the new value for the current key to make the total 100
    st.session_state[key] = max(0, 100 - total_excluding_current)

# Sidebar for weight adjustments
st.sidebar.header('Set Weights for Parameters')
for key in initial_weights:
    st.sidebar.slider(f'Weight for {key}', 0, 100, st.session_state[key], key=key, on_change=adjust_weights, args=(key,))

# Display pie chart for weight distribution
weight_data = pd.DataFrame(list(st.session_state.items()), columns=['Parameter', 'Weight'])
fig = px.pie(weight_data, values='Weight', names='Parameter', title='Weight Distribution')
st.plotly_chart(fig)

# Main section for parameter inputs
st.header('Adjust Construction Parameters')

# Design Efficiency
st.subheader('1. Design Efficiency')
design_efficiency = st.slider('Design Efficiency', 0, 100, 50)
construction_type = st.selectbox('Type of Construction', ['Long-span', 'Short-span'])
non_parking_areas = st.checkbox('Include Non-parking Areas')
double_loaded_bays = st.checkbox('Include Double-loaded Bays with Perimeter Parking')
parking_ramps = st.radio('Parking Ramps Type', ['Standard Ramps', 'Express Ramps'])
parking_controls = st.checkbox('Include Parking Controls')
parking_footprint = st.slider('Parking Footprint', 0, 100, 50)

# On grade and above grade vs subterranean
st.subheader('2. On Grade and Above Grade vs Subterranean')
subterranean = st.checkbox('Include Subterranean Levels')

# Design Redundancy
st.subheader('3. Design Redundancy')
design_redundancy = st.slider('Design Redundancy', 0, 100, 50)

# Soil conditions
st.subheader('4. Soil Conditions')
soil_condition = st.slider('Soil Condition Quality', 0, 100, 50)

# Additional parameters
st.subheader('Additional Parameters')
location_cost = st.slider('Cost of Labor and Materials by Location', 0, 100, 50)
seismic_zone = st.slider('Seismic Zone Impact', 0, 100, 50)
site_access = st.text_input('Site Access Difficulty')
number_of_bidders = st.number_input('Number of Bidders', min_value=1, max_value=20, value=5)
economic_factors = st.slider('Economic Factors', 0, 100, 50)
type_of_delivery = st.selectbox('Type of Delivery', ['Design-Bid-Build', 'Design-Build'])

# Façade treatment & Finishes
st.subheader('11. Façade Treatment & Finishes')
facade_treatment = st.text_input('Describe Façade Treatment and Finishes')

# Ventilation
st.subheader('12. Ventilation')
ventilation_type = st.radio('Ventilation Type', ['Open', 'Closed'])

# Sprinkler or standpipe
st.subheader('13. Sprinkler or Standpipe')
sprinkler_system = st.checkbox('Include Sprinkler System')

# Utilities
st.subheader('14. Utilities')
utilities_inclusion = st.checkbox('Include Utilities')

# Construction method
st.subheader('15. Construction Method')
construction_method = st.radio('Construction Method', ['Cast-in-place', 'Precast'])

# Structural system
st.subheader('16. Structural System')
structural_system = st.radio('Structural System', ['Moment Frame', 'Shear Walls'])

# Time (cost escalation)
st.subheader('17. Time (Cost Escalation)')
time_cost_escalation = st.slider('Time Cost Escalation', 0, 100, 50)

# Button to perform calculation
if st.button('Calculate Costs'):
    base_cost = 10000000  # Base cost for a standard multi-level parking garage
    
    # Get parameters and their corresponding weights from session state
    parameters = {
        'Design Efficiency': design_efficiency,
        'Type of Construction': 1 if construction_type == 'Long-span' else 0,
        'Non-parking Areas': 1 if non_parking_areas else 0,
        'Double-loaded Bays': 1 if double_loaded_bays else 0,
        'Parking Ramps': 1 if parking_ramps == 'Express Ramps' else 0,
        'Parking Controls': 1 if parking_controls else 0,
        'Parking Footprint': parking_footprint,
        'Subterranean Levels': 1 if subterranean else 0,
        'Design Redundancy': design_redundancy,
        'Soil Condition': soil_condition,
        'Location Cost': location_cost,
        'Seismic Zone': seismic_zone,
        'Site Access': 50,  # Default value for text input
        'Number of Bidders': number_of_bidders * 5,  # Scale to 100
        'Economic Factors': economic_factors,
        'Type of Delivery': 1 if type_of_delivery == 'Design-Build' else 0,
        'Façade Treatment': 50,  # Default value for text input
        'Ventilation Type': 1 if ventilation_type == 'Closed' else 0,
        'Sprinkler System': 1 if sprinkler_system else 0,
        'Utilities Inclusion': 1 if utilities_inclusion else 0,
        'Construction Method': 1 if construction_method == 'Precast' else 0,
        'Structural System': 1 if structural_system == 'Shear Walls' else 0,
        'Time Cost Escalation': time_cost_escalation
    }
    
    # Calculate adjusted cost using parameters and their weights from session state
    adjusted_cost = base_cost
    for key in initial_weights:
        weight = st.session_state[key]
        param_value = parameters[key]
        adjusted_cost *= (1 + (param_value/100) * (weight/100))

    st.write(f"Estimated Construction Cost: ${adjusted_cost:,.2f}")

    # Cost breakdown chart
    cost_factors = ['Base Cost']
    amounts = [base_cost]
    for key in initial_weights:
        cost_factors.append(key)
        weight = st.session_state[key]
        param_value = parameters[key]
        amounts.append(base_cost * (param_value/100) * (weight/100))
    
    data = {'Cost Factors': cost_factors, 'Amount': amounts}
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index('Cost Factors')) 
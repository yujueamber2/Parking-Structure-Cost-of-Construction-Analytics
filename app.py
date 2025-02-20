# Add to imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import joblib
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from historical_data import HISTORICAL_PROJECTS

# Database Setup
Base = declarative_base()

class CostScenario(Base):
    __tablename__ = 'scenarios'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    parameters = Column(JSON)
    weights = Column(JSON)
    total_cost = Column(Float)
    timestamp = Column(DateTime)

engine = create_engine('sqlite:///cost_data.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# App Configuration
st.set_page_config(page_title="Parking Cost Analyzer", layout="wide")

# Initialize Parameters
def initialize_parameters():
    return {
        "Design Efficiency": {"weight": 0.15, "value": 3},
        "Parking Standards": {"weight": 0.1, "value": 4},
        "Construction Type": {"weight": 0.12, "value": 2},
        "Structure Type": {"weight": 0.2, "value": 1},
        "Levels": {"weight": 0.15, "value": 3},
        "Material Type": {"weight": 0.18, "value": 1},
        "Site Work": {"weight": 0.1, "value": 2},
        "Facade Quality": {"weight": 0.08, "value": 1},
        "PV Roof": {"weight": 0.05, "value": 0},
        "Non-Parking Uses": {"weight": 0.07, "value": 0},
    }

# Cost Calculation
def calculate_cost(parameters):
    base_cost = 1000000
    cost_factors = sum(p["weight"] * p["value"] for p in parameters.values())
    return base_cost * (1 + cost_factors)

# Database Functions
def get_saved_scenarios():
    with Session() as session:
        return session.query(CostScenario).all()

def get_scenario_versions(name):
    with Session() as session:
        return session.query(CostScenario)\
            .filter(CostScenario.name == name)\
            .order_by(CostScenario.timestamp.desc())\
            .all()

# ML Functions
def preprocess_data(df):
    le = LabelEncoder()
    categorical_cols = ['Construction Type', 'Ramping', 'Ventilation', 
                       'Site Work', 'Facade Quality', 'Non-Parking Uses']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    bool_cols = ['PV Roof']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    return df

def train_cost_predictor():
    with Session() as session:
        user_data = session.query(CostScenario).all()
    
    user_df = pd.DataFrame([{
        **scenario.parameters,
        **scenario.weights,
        'cost_per_space': scenario.total_cost / 1000
    } for scenario in user_data]) if user_data else pd.DataFrame()
    
    historical_df = pd.DataFrame(HISTORICAL_PROJECTS)
    historical_df = preprocess_data(historical_df)
    
    combined_df = pd.concat([user_df, historical_df], ignore_index=True)
    
    if len(combined_df) < 5:
        st.warning("Need at least 5 data points for reliable predictions")
        return None, None, None
    
    X = combined_df.drop(['name', 'location', 'completion_date', 'cost_per_space'], 
                        axis=1, errors='ignore')
    y = combined_df['cost_per_space']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, 'cost_predictor.pkl')
    return model, X_test, y_test

# Historical Data Tab
def show_historical_reference():
    st.header("📚 Historical Project Reference")
    
    selected_project = st.selectbox("Choose Project", 
                                  [p["name"] for p in HISTORICAL_PROJECTS])
    
    project = next(p for p in HISTORICAL_PROJECTS if p["name"] == selected_project)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Basic Info")
        st.markdown(f"""
        - **Location:** {project['location']}
        - **Completion Date:** {project['completion_date']}
        - **Total Square Footage:** {project['sq_ft']:,} SF
        - **Cost Per Space:** ${project['cost_per_space']:,.2f}
        """)
    
    with col2:
        st.subheader("Technical Details")
        st.markdown(f"""
        - **Levels:** {project['levels']}
        - **Structural System:** {project['structural_system']}
        - **Ramping Type:** {project['ramping']}
        - **Ventilation:** {project['ventilation']}
        - **Parking Spaces:** {project['spaces']:,}
        """)
    
    st.subheader("Comparative Analysis")
    fig = px.bar(pd.DataFrame(HISTORICAL_PROJECTS), 
                x='name', y='cost_per_space',
                title="Cost Per Space Comparison")
    st.plotly_chart(fig, use_container_width=True)

# Main App
def main():
    st.title("🏗️ Parking Structure Cost Analyzer")
    
    tab1, tab2, tab3 = st.tabs(["Cost Analyzer", "Historical Reference", "ML Insights"])
    
    with tab1:
        parameters = initialize_parameters()
        
        with st.sidebar:
            st.header("Project Parameters")
            
            total_weight = 0
            for param in parameters:
                parameters[param]["weight"] = st.slider(
                    f"Weight for {param}",
                    0.0, 1.0,
                    value=parameters[param]["weight"],
                    key=f"weight_{param}"
                )
                total_weight += parameters[param]["weight"]
            
            if abs(total_weight - 1.0) > 0.01:
                st.error(f"Total weights must sum to 100%! Current sum: {total_weight*100:.2f}%")
            
            with st.expander("💾 Scenario Management"):
                scenario_name = st.text_input("Scenario Name")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Scenario"):
                        with Session() as session:
                            new_scenario = CostScenario(
                                name=scenario_name,
                                parameters={k: v["value"] for k, v in parameters.items()},
                                weights={k: v["weight"] for k, v in parameters.items()},
                                total_cost=calculate_cost(parameters),
                                timestamp=datetime.now()
                            )
                            session.add(new_scenario)
                            session.commit()
                            st.success("Scenario saved!")
                
                with col2:
                    scenarios = get_saved_scenarios()
                    scenario_names = list({s.name for s in scenarios})
                    selected_scenario = st.selectbox("Load Scenario", scenario_names)
                    
                    if st.button("Load Selected"):
                        with Session() as session:
                            scenario = session.query(CostScenario)\
                                .filter(CostScenario.name == selected_scenario)\
                                .order_by(CostScenario.timestamp.desc())\
                                .first()
                            for param in parameters:
                                parameters[param]["weight"] = scenario.weights[param]
                                parameters[param]["value"] = scenario.parameters[param]
                            st.experimental_rerun()
        
        current_cost = calculate_cost(parameters)
        st.subheader(f"Estimated Cost: ${current_cost:,.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cost_data = pd.DataFrame({
                "Parameter": parameters.keys(),
                "Impact": [p["weight"] * p["value"] for p in parameters.values()]
            })
            fig = px.bar(cost_data, x="Parameter", y="Impact", title="Cost Impact Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🤖 ML Cost Prediction")
            
            if st.button("Train/Retrain Prediction Model"):
                model, X_test, y_test = train_cost_predictor()
                if model:
                    predictions = model.predict(X_test)
                    fig = px.scatter(x=y_test, y=predictions, 
                                   labels={'x': 'Actual', 'y': 'Predicted'},
                                   title="Model Accuracy")
                    st.plotly_chart(fig)
            
            if os.path.exists('cost_predictor.pkl'):
                model = joblib.load('cost_predictor.pkl')
                current_features = pd.DataFrame([{
                    **{k: v["value"] for k, v in parameters.items()},
                    **{k: v["weight"] for k, v in parameters.items()}
                }])
                try:
                    prediction = model.predict(current_features)
                    st.metric("ML Predicted Cost", f"${prediction[0]:,.2f}")
                except Exception as e:
                    st.error("Model needs retraining with current parameters")
    
    with tab2:
        show_historical_reference()
    
    with tab3:
        st.header("🤖 Machine Learning Insights")
        
        if os.path.exists('cost_predictor.pkl'):
            model = joblib.load('cost_predictor.pkl')
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': model.feature_names_in_,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance Analysis")
                fig = px.bar(importance_df, x='feature', y='importance',
                            title="Impact Factors on Parking Costs")
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Training Data Composition")
        source_counts = pd.DataFrame({
            'Source': ['User Scenarios', 'Historical Projects'],
            'Count': [len(get_saved_scenarios()), len(HISTORICAL_PROJECTS)]
        })
        fig = px.pie(source_counts, names='Source', values='Count',
                    title="Data Source Distribution")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
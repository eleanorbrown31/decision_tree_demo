import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

# Set up the app title
st.title("Live Decision Tree Learning")
st.write("Watch how a decision tree learns from audience data!")

# Initialize session state to store our data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({
        'coffee_tea': [],
        'sleep_hours': [],
        'wfh_office': [],
        'morning_night': []
    })

# Create form to collect data
with st.form("data_collection"):
    st.write("Add your data to train the model:")
    
    coffee_tea = st.radio("Do you prefer coffee or tea?", ["Coffee", "Tea"])
    sleep_hours = st.slider("How many hours do you sleep per night?", 4, 10, 7)
    wfh_office = st.radio("Do you prefer working from home or office?", ["Home", "Office"])
    morning_night = st.radio("Are you a morning person or night owl?", ["Morning", "Night"])
    
    submit_button = st.form_submit_button("Add my data")
    
    if submit_button:
        # Add the new data point
        new_data = pd.DataFrame({
            'coffee_tea': [0 if coffee_tea == "Coffee" else 1],
            'sleep_hours': [sleep_hours],
            'wfh_office': [0 if wfh_office == "Home" else 1],
            'morning_night': [0 if morning_night == "Morning" else 1]
        })
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
        st.success("Data added!")

# Display current dataset
st.subheader("Current Dataset")
if not st.session_state.data.empty:
    display_df = st.session_state.data.copy()
    display_df['coffee_tea'] = display_df['coffee_tea'].map({0: 'Coffee', 1: 'Tea'})
    display_df['wfh_office'] = display_df['wfh_office'].map({0: 'Home', 1: 'Office'})
    display_df['morning_night'] = display_df['morning_night'].map({0: 'Morning', 1: 'Night'})
    st.write(display_df)
else:
    st.info("No data collected yet. Please add some data points.")

# Train the model if we have data
if not st.session_state.data.empty and len(st.session_state.data) >= 3:
    st.subheader("Decision Tree Model")
    
    # Prepare data
    X = st.session_state.data[['coffee_tea', 'sleep_hours', 'wfh_office']]
    y = st.session_state.data['morning_night']
    
    # Train decision tree with adjusted parameters
    clf = tree.DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=1,  # Allow leaves with just one sample (default)
        min_samples_split=2, # Default value, split with as few as 2 samples
        criterion='entropy'   # Use entropy instead of gini for more nuanced splits
    )
    clf = clf.fit(X, y)
    
    # Visualize the tree
    dot_data = export_graphviz(
        clf, 
        out_file=None,
        feature_names=['Coffee/Tea', 'Sleep Hours', 'WFH/Office'],
        class_names=['Morning Person', 'Night Owl'],
        filled=True, 
        rounded=True
    )
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(dot_data)
    
    # Make predictions
    st.subheader("Try a prediction")
    
    with st.form("prediction"):
        pred_coffee_tea = st.radio("Coffee or Tea?", ["Coffee", "Tea"], key="pred_coffee")
        pred_sleep_hours = st.slider("Sleep hours?", 4, 10, 7, key="pred_sleep")
        pred_wfh_office = st.radio("Home or Office?", ["Home", "Office"], key="pred_wfh")
        
        predict_button = st.form_submit_button("Predict")
        
        if predict_button:
            # Convert inputs to model format
            pred_input = np.array([[
                0 if pred_coffee_tea == "Coffee" else 1,
                pred_sleep_hours,
                0 if pred_wfh_office == "Home" else 1
            ]])
            
            # Make prediction
            prediction = clf.predict(pred_input)[0]
            prediction_prob = clf.predict_proba(pred_input)[0]
            
            # Convert prediction to integer to ensure it's a valid index
            prediction_idx = int(prediction)
            
            # Get the confidence value
            confidence = prediction_prob[prediction_idx]
            
            # Display result with more informative confidence message
            confidence_msg = ""
            if confidence > 0.95:
                confidence_msg = "(Very confident)"
            elif confidence > 0.75:
                confidence_msg = "(Fairly confident)"
            elif confidence > 0.6:
                confidence_msg = "(Somewhat confident)"
            else:
                confidence_msg = "(Uncertain)"
                
            if prediction_idx == 0:
                st.success(f"Prediction: Morning Person - {confidence_msg} ({prediction_prob[0]:.2f})")
            else:
                st.success(f"Prediction: Night Owl - {confidence_msg} ({prediction_prob[1]:.2f})")
else:
    st.warning("Need at least 3 data points to train the model.")
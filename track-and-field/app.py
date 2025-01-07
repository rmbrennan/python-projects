import streamlit as st
import pandas as pd

# Sample data for events and athletes
events = {
    'Mens 100m': ['Noah Lyles', 'Kishane Thompson'],
    'Mens 200m': ['Noah Lyles', 'Andre De Grasse'],
    'Womens 100m': ['Shacarri Richardson', 'Julien Alfred'],
    'Shot Put': ['Athlete 7', 'Athlete 8'],
    'Long Jump': ['Athlete 9', 'Athlete 10']
}

# Initialize or load predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Title of the app
st.title("Olympic Medal Prediction App")

# User name input
user_name = st.text_input("Enter your name:")

# Drop down lists for each event
for event, athletes in events.items():
    st.subheader(event)
    selected_athlete = st.selectbox(
        f"Select the athlete you think will win a medal in {event}:",
        options=[""] + athletes,
        key=event
    )
    if selected_athlete:
        st.session_state.predictions[event] = selected_athlete

# Submit button
if st.button("Submit Predictions"):
    if user_name:
        # Save the predictions (in a real app, save to a database)
        st.session_state.predictions['user'] = user_name
        st.success(f"Predictions submitted for {user_name}!")
        st.write(st.session_state.predictions)
    else:
        st.error("Please enter your name before submitting.")

# Display predictions
if st.checkbox("Show predictions"):
    st.write(st.session_state.predictions)
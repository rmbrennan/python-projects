import pandas as pd
import streamlit as st
import logging
import asyncio
from main import setup_agent_pipeline  # Assuming main.py is in the same directory
from agents.data_scraper import DataScraperAgent
from agents.team_optimizer import TeamOptimizer  # Import the TeamOptimizer class

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize session state to store pipeline results
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# Streamlit UI
st.title("FPL Team Optimizer")
st.markdown("### A Streamlit app for optimizing your Fantasy Premier League team")

# Add input for user team ID
user_team_id = st.sidebar.text_input("Enter your FPL Team ID:", value="6256406")  # Default to a valid team ID

# Add a button to run the optimizer
if st.button("Run Team Optimizer"):
    st.info("Running the team optimizer...")
    
    # Load players data using the DataScraperAgent
    data_scraper = DataScraperAgent(user_team_id=int(user_team_id))
    data = asyncio.run(data_scraper.process())
    all_players = data['all_players']
    
    # Initialize the optimizer
    optimizer = TeamOptimizer(all_players)
    
    # Run the optimization
    best_team = optimizer.create_optimal_team(iterations=1000)
    
    # Store the results in session state
    st.session_state.best_team = best_team
    st.success("Team optimization completed!")

# Display the best team if available
if "best_team" in st.session_state:
    starting_11, bench = st.session_state.best_team
    
    st.subheader("Best Starting 11")
    st.write("Here are the optimal players for your starting 11:")
    
    # Display the starting 11 as a table
    st.dataframe(starting_11[['name', 'position', 'expected_points']])
    
    st.subheader("Bench")
    st.write("Here are the players on your bench:")
    
    # Display the bench as a table
    st.dataframe(bench[['name', 'position', 'expected_points']])
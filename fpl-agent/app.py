import pandas as pd
import streamlit as st
import logging
from main import setup_agent_pipeline  # Assuming main.py is in the same directory

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_pipeline(user_team_id: int):
    try:
        st.info("Starting the agent pipeline...")
        agents = setup_agent_pipeline(user_team_id)  # Pass user_team_id here
        results = {}
        
        # Validate and process agents
        for agent in agents:
            if agent.validate():
                result = agent.process()
                results[agent.name] = result
                st.success(f"{agent.name} completed processing.")
                
                # Display data for DataScraperAgent
                if agent.name == "DataScraper":
                    st.subheader("DataScraper Results")
                    
                    # Get the list of available datasets
                    datasets = list(result.keys())
                    
                    # Allow the user to select which datasets to display
                    selected_datasets = st.sidebar.multiselect(
                        "Select datasets to display",
                        datasets,
                        default=datasets
                    )
                    
                    # Allow the user to reorder the datasets
                    reordered_datasets = st.sidebar.selectbox(
                        "Select a dataset to move",
                        selected_datasets,
                        key="dataset_to_move"
                    )
                    
                    # Move the selected dataset to the top of the list
                    if st.sidebar.button("Move to top"):
                        selected_datasets.remove(reordered_datasets)
                        selected_datasets.insert(0, reordered_datasets)
                    
                    # Display the selected datasets
                    for dataset in selected_datasets:
                        if st.sidebar.checkbox(dataset, key=dataset):
                            st.write(f"### {dataset.capitalize()} Data")
                            if isinstance(result[dataset], pd.DataFrame):
                                st.dataframe(result[dataset])
                            elif isinstance(result[dataset], (list, dict)):
                                st.json(result[dataset])
                            else:
                                st.write(result[dataset])
            else:
                st.error(f"Validation failed for {agent.name}.")
        
        return results
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        st.error(f"Pipeline failed: {str(e)}")

# Streamlit UI
st.title("Agent Pipeline Runner")

# Add input for user team ID
user_team_id = st.sidebar.text_input("Enter your FPL Team ID:", value="6256406")  # Default to a valid team ID

# Run the pipeline
results = run_pipeline(user_team_id)
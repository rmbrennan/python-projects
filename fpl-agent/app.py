import pandas as pd
import streamlit as st
import logging
import asyncio
from main import setup_agent_pipeline  # Assuming main.py is in the same directory

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize session state to store pipeline results
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

async def run_pipeline(user_team_id: int):
    try:
        st.info("Starting the agent pipeline...")
        agents = setup_agent_pipeline(user_team_id)
        results = {}

        # Validate and process agents
        for agent in agents:
            if agent.validate():
                result = await agent.process()  # Await the coroutine
                results[agent.name] = result
                st.success(f"{agent.name} completed processing.")
            else:
                st.error(f"Validation failed for {agent.name}.")

        return results
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        st.error(f"Pipeline failed: {str(e)}")
        return None

# Streamlit UI
st.title("Agent Pipeline Runner")
st.markdown("### A Streamlit app for running agent pipelines")

# Add input for user team ID
user_team_id = st.sidebar.text_input("Enter your FPL Team ID:", value="6256406")  # Default to a valid team ID

# Run the pipeline only once when the button is clicked
if st.button("Run Pipeline"):
    st.session_state.pipeline_results = asyncio.run(run_pipeline(int(user_team_id)))

# Check if pipeline results are available in session state
if st.session_state.pipeline_results:
    results = st.session_state.pipeline_results

    # Display data for DataScraperAgent
    if "DataScraper" in results:
        st.subheader("DataScraper Results")

        # Get the list of available datasets
        datasets = list(results["DataScraper"].keys())

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
                if isinstance(results["DataScraper"][dataset], pd.DataFrame):
                    st.dataframe(results["DataScraper"][dataset])
                elif isinstance(results["DataScraper"][dataset], (list, dict)):
                    st.json(results["DataScraper"][dataset])
                else:
                    st.write(results["DataScraper"][dataset])

# Add a collapsible section for documentation
with st.sidebar.expander("Documentation"):
    st.markdown("### How to use this app")
    st.markdown("1. Enter your FPL Team ID in the sidebar")
    st.markdown("2. Click the 'Run Pipeline' button to start the pipeline")
    st.markdown("3. Select which datasets to display in the sidebar")
    st.markdown("4. Use the checkboxes to toggle dataset display")
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
                    
                    # Display all datasets
                    for key, value in result.items():
                        st.write(f"### {key.capitalize()} Data")
                        if isinstance(value, pd.DataFrame):
                            st.dataframe(value)
                        elif isinstance(value, (list, dict)):
                            st.json(value)
                        else:
                            st.write(value)
            else:
                st.error(f"Validation failed for {agent.name}.")
        
        return results
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        st.error(f"Pipeline failed: {str(e)}")
    
# Streamlit UI
st.title("Agent Pipeline Runner")

# Add input for user team ID
user_team_id = st.text_input("Enter your FPL Team ID:", value="6256406")  # Default to a valid team ID

if st.button("Run Pipeline"):
    results = run_pipeline(user_team_id)
    
    if results:
        st.subheader("Results:")
        for agent_name, result in results.items():
            if st.checkbox(f"Show output for {agent_name}", value=False):
                st.write(f"{agent_name} output: {result}")

# if st.button("Reset"):
#     st.experimental_rerun()  # Reset the app state
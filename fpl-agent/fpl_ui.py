import os
import streamlit as st
import asyncio
import pandas as pd
from fpl_agents import FPLAgent, DataScraperAgent, TeamOptimizationAgent,  TransferAgent, CaptainAgent # Add more agents here as needed

# Initialize available agents
available_agents = {
    "Data Scraper": DataScraperAgent(),
    "Team Optimizer": TeamOptimizationAgent(),
    "Transfer Agent": TransferAgent(),
    "Captain Agent": CaptainAgent(),
    # Add more agents like "Performance Analyzer": PerformanceAnalyzerAgent(), etc.
}

# Function to run agent processing
async def run_agent(agent: FPLAgent, input_data: dict):
    try:
        return await agent.process(input_data)
    except Exception as e:
        return {"error": str(e)}

# Define Streamlit UI
st.title("Fantasy Premier League Agent Explorer")
st.sidebar.header("Agent Selection")

# Select an agent
selected_agent_name = st.sidebar.selectbox("Choose an Agent", list(available_agents.keys()))
selected_agent = available_agents[selected_agent_name]

# Agent documentation
st.subheader(f"Agent: {selected_agent_name}")
st.write(f"**Description:** {selected_agent.__doc__ or 'No description available.'}")

# Input/Output Interaction
st.subheader("Agent Interaction")

# Input Fields
input_section = st.expander("Agent Input Configuration", expanded=True)
with input_section:
    st.write("Configure the inputs for the agent (if required).")
    input_data = {}  # You can add specific fields for user inputs here.

# Run Agent Button
if st.button("Run Agent"):
    st.info("Processing... Please wait.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        output = loop.run_until_complete(run_agent(selected_agent, input_data))
        st.success("Processing Complete!")
        
        # Display Outputs
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "recommended_team":
                    st.subheader("Recommended Team")
                    recommended_team_df = pd.DataFrame(value)
                    st.dataframe(recommended_team_df)
                else:
                    st.subheader(f"Output: {key}")
                    if isinstance(value, pd.DataFrame):
                        st.dataframe(value)
                    elif isinstance(value, (list, dict)):
                        st.json(value)
                    else:
                        st.write(value)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        loop.close()

# Footer
st.sidebar.markdown("Developed by Robbie")
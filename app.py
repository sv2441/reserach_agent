import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent, load_tools
# from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
import streamlit as st
import pandas as pd 

st.set_page_config(
    page_title="Prodago Research Agent", page_icon="🚀",
    layout="wide",
)

st.title("Welcome to Prodago Research Agent 🚀")

wikipedia = WikipediaAPIWrapper()
wikipedia.run('Langchain')

python_repl = PythonREPL()
search = DuckDuckGoSearchRun()

load_dotenv()

llm = OpenAI(temperature=0.6, streaming=True, openai_api_key=st.secrets["OPENAI_API_KEY"])

from langchain.agents import Tool

tools = [
    Tool(
        name="python repl",
        func=python_repl.run,
        description="useful for when you need to use python to answer a question. You should input python code"
    )
]

wikipedia_tool = Tool(
    name='wikipedia',
    func=wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func=search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

tools.append(duckduckgo_tool)
tools.append(wikipedia_tool)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True
)

# File Upload and Processing
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    results_df = pd.DataFrame(columns=['prompt', 'results','Competitor','Topic','Week'])  # Create a new DataFrame to store results
    for index, row in df.iterrows():
        prompt = row['Prompt']
        response = agent.run(prompt)
        results_df = results_df.append({'prompt': prompt, 'results': response}, ignore_index=True)

    # Display the results DataFrame
    st.write("Results:")
    st.dataframe(results_df)

    # Download button for results DataFrame
    st.download_button(
        label="Download Results CSV",
        data=results_df.to_csv(index=False),
        file_name="research_results.csv",
        mime="text/csv"
    )

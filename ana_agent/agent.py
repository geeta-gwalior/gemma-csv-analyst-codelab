from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import pandas as pd
import os


# =========================================================
# 1. USER-DEFINED TOOL
# =========================================================

def read_csv() -> str:
    """
    Read the st.csv file from the same folder as agent.py.
    """

    file_path = os.path.join(
        os.path.dirname(__file__),
        "st.csv"
    )

    try:
        df = pd.read_csv(file_path)

        return df.to_string(index=False)

    except Exception as e:
        return f"Error reading CSV: {str(e)}"


# =========================================================
# 2. CSV ANALYST AGENT
# =========================================================

csv_analyst = Agent(
    name="csv_analyst",

    model=LiteLlm(
        model="ollama_chat/gemma4:e4b",
        temperature=0
    ),

    description="""
    Specialist agent for analyzing CSV data.
    """,

    instruction="""
    You are a CSV Data Analyst.

    Your job is to analyze the data from st.csv.

    ALWAYS use the read_csv tool before answering
    any question about the CSV.

    You can answer questions such as:

    - What is the average age?
    - What is the maximum age?
    - What is the minimum age?
    - How many students are there?
    - What is the average salary?
    - Which student has the highest salary?
    - Compare two students.
    - Find important patterns in the data.

    Use only the information returned by the
    read_csv tool.

    Do not invent information.

    After analyzing the data, provide a clear
    answer to the user.
    """,

    tools=[read_csv],
)


# =========================================================
# 3. CSV REPORT AGENT
# =========================================================

csv_report_agent = Agent(
    name="csv_report_agent",

    model=LiteLlm(
        model="ollama_chat/gemma4:e4b",
        temperature=0
    ),

    description="""
    Specialist agent for creating summaries and
    reports from CSV data.
    """,

    instruction="""
    You are a CSV Report Specialist.

    Your job is to create summaries and reports
    from the st.csv file.

    ALWAYS use the read_csv tool first.

    After reading the data:

    1. Understand the columns.
    2. Identify important information.
    3. Find useful statistics.
    4. Highlight important findings.
    5. Create a clear summary.
    6. Provide a report with key insights.
    7.create some charts or visualizations if possible.

    Use only information from the CSV.

    Do not invent information.
    """,

    tools=[read_csv],
)


# =========================================================
# 4. ROOT / ORCHESTRATOR AGENT
# =========================================================

root_agent = Agent(
    name="csv_orchestrator",

    model=LiteLlm(
        model="ollama_chat/gemma4:e4b",
        temperature=0
    ),

    description="""
    Main orchestrator for CSV-related questions.
    Delegates tasks to specialist agents.
    """,

    instruction="""
    You are the CSV Orchestrator.

    You have two specialist agents:

    1. csv_analyst
       Use this agent for:
       - calculations
       - averages
       - maximum/minimum
       - counts
       - comparisons
       - data analysis

    2. csv_report_agent
       Use this agent for:
       - summaries
       - reports
       - important findings
       - overall analysis

    Do NOT analyze the CSV yourself.

    Delegate the request to the appropriate
    specialist agent.

    Examples:

    User:
    "What is the average age?"
    -> csv_analyst

    User:
    "Which student has the highest salary?"
    -> csv_analyst

    User:
    "How many students are there?"
    -> csv_analyst

    User:
    "Give me a summary of the CSV."
    -> csv_report_agent

    User:
    "Create a report from the student data."
    -> csv_report_agent

    After the specialist completes the task,
    return its answer to the user.
    """,

    sub_agents=[
        csv_analyst,
        csv_report_agent
    ],
)
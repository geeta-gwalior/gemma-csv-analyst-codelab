# Build a CSV Analyst with Google ADK and Gemma (Local, Free)

In this codelab you will build a small team of AI agents that read a CSV file and answer questions about it. Everything runs on your own laptop. No cloud, no API key, no cost.

**Time:** about 30 minutes
**Level:** Beginner

## What you will build

```
You ask a question
        |
        v
csv_orchestrator  (the manager)
   |            |
   v            v
csv_analyst   csv_report_agent
(numbers)     (summary + charts)
   |            |
   +-- read_csv tool --> st.csv
```

## What you will learn

- What an **agent**, a **tool** and a **sub-agent** are
- How to run **Gemma** locally with **Ollama**
- How to connect Gemma to **Google ADK** using `LiteLlm`
- How one manager agent can send work to specialist agents

## What is Gemma?

**Gemma** is a family of open AI models from Google DeepMind. It is built using the same research as Google's Gemini models, but you can download Gemma and run it yourself.

Gemma comes in different sizes. Small ones (like `gemma4:e4b` used here) run on a normal laptop. Bigger ones need a stronger GPU or the cloud.

### Why is Gemma useful?

| Benefit | What it means for you |
|---------|-----------------------|
| **Free to run** | No API bill. Download once and use it. |
| **Private** | Your data stays on your machine. Good for company or student data. |
| **Works offline** | No internet needed after the download. |
| **Tool calling** | Gemma can call your Python functions, which is what agents need. |
| **Flexible** | Run it locally with Ollama, or deploy it on Google Cloud when you need scale. |

### Gemma or Gemini?

- **Gemini**: Google's most powerful models, used through an API in the cloud.
- **Gemma**: smaller open models that you can run and control yourself.

In this codelab we use Gemma because it is free, private and easy to run in a workshop.

> Note: a small local model is less powerful than a big cloud model. That is why we keep instructions short and clear, and use `temperature=0`.

## Basics in 1 minute

| Word | Simple meaning |
|------|----------------|
| **LLM** | The "brain" that understands and writes text. Here it is Gemma. |
| **Agent** | An LLM with a job (instructions) and, optionally, tools. |
| **Tool** | A normal Python function the agent can call. Example: read a file. |
| **Sub-agent** | A specialist agent that a manager agent can hand work to. |
| **ADK** | Agent Development Kit. Google's Python framework to build agents. |
| **Ollama** | A free app that runs open models like Gemma on your computer. |

---

## Step 1: Install the requirements

You need Python 3.10 or higher.

1. Install Ollama from https://ollama.com
2. Download the Gemma model:

```bash
ollama pull gemma4:e4b
```

3. Install the Python packages:

```bash
pip install "google-adk[extensions]" pandas matplotlib
```

`[extensions]` is needed because ADK talks to Ollama through LiteLLM.

## Step 2: Create the project folder

```
my_project/
  csv_agent/
    __init__.py
    agent.py
    st.csv
    .env
```

**`__init__.py`** (one line, so ADK can find your agent):

```python
from . import agent
```

**`.env`** (tells ADK where Ollama is running):

```
OLLAMA_API_BASE=http://localhost:11434
```

**`st.csv`** (sample data, you can use your own):

```csv
name,age,city,salary
Asha,21,Gwalior,30000
Ravi,24,Indore,45000
Meera,22,Bhopal,38000
Kabir,26,Delhi,60000
Sana,23,Pune,42000
```

## Step 3: Write the tool

A tool is just a Python function. The **docstring is important**: the model reads it to know what the tool does.

Open `agent.py` and start with this:

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import pandas as pd
import os


def read_csv() -> str:
    """Read the st.csv file from the same folder as agent.py."""
    file_path = os.path.join(os.path.dirname(__file__), "st.csv")
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading CSV: {str(e)}"
```

**Why:** the model cannot open files by itself. The tool gives it the data.

## Step 4: Create the first agent (the analyst)

```python
csv_analyst = Agent(
    name="csv_analyst",
    model=LiteLlm(model="ollama_chat/gemma4:e4b", temperature=0),
    description="Specialist agent for analyzing CSV data.",
    instruction="""
    You are a CSV Data Analyst.
    ALWAYS use the read_csv tool before answering any question about the CSV.
    Answer questions like averages, maximum, minimum, counts and comparisons.
    Use only the information returned by the read_csv tool.
    Do not invent information.
    """,
    tools=[read_csv],
)
```

Two things to notice:

- **`ollama_chat/...`** : always use `ollama_chat`, not `ollama`. The plain `ollama` prefix can cause tool-call loops.
- **`temperature=0`** : makes answers steady and less random, which is good for numbers.

## Step 5: Test one agent quickly (optional but recommended)

Temporarily add this at the bottom of `agent.py`:

```python
root_agent = csv_analyst
```

Run from `my_project/`:

```bash
adk web
```

Open http://localhost:8000, pick `csv_agent`, and ask:

> What is the average age?

If you get the right answer, your setup works. Remove the temporary line before the next step.

## Step 6: Add the chart tool and the report agent

First the chart tool. Put this near `read_csv`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_chart(column: str, chart_type: str = "bar") -> str:
    """Create a chart for one CSV column and save it as a PNG file.
    chart_type can be 'bar', 'hist' or 'line'."""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "st.csv"))
    if column not in df.columns:
        return f"Column not found. Available columns: {list(df.columns)}"
    out = os.path.join(os.path.dirname(__file__), f"{column}_{chart_type}.png")
    fig, ax = plt.subplots()
    if chart_type == "hist":
        df[column].plot.hist(ax=ax)
    elif chart_type == "line":
        df[column].plot.line(ax=ax)
    else:
        df[column].value_counts().plot.bar(ax=ax)
    ax.set_title(f"{column} ({chart_type})")
    fig.savefig(out)
    plt.close(fig)
    return f"Chart saved at {out}"
```

Now the report agent:

```python
csv_report_agent = Agent(
    name="csv_report_agent",
    model=LiteLlm(model="ollama_chat/gemma4:e4b", temperature=0),
    description="Specialist agent for creating summaries and reports from CSV data.",
    instruction="""
    You are a CSV Report Specialist.
    ALWAYS use the read_csv tool first.
    Then:
    1. Understand the columns.
    2. Find useful statistics.
    3. Write a short summary with key findings.
    4. Use the create_chart tool to make one chart for an important column.
    Use only information from the CSV. Do not invent information.
    """,
    tools=[read_csv, create_chart],
)
```

**Why two agents?** Small, focused agents are more reliable than one agent that does everything.

## Step 7: Create the manager (root agent)

ADK looks for a variable named **`root_agent`**. This is the entry point.

```python
root_agent = Agent(
    name="csv_orchestrator",
    model=LiteLlm(model="ollama_chat/gemma4:e4b", temperature=0),
    description="Main manager for CSV questions. Delegates to specialists.",
    instruction="""
    You are the CSV Orchestrator. You have two specialists:

    1. csv_analyst: calculations, averages, max/min, counts, comparisons.
    2. csv_report_agent: summaries, reports, key findings, charts.

    Do NOT analyze the CSV yourself. Send the request to the right specialist.

    Examples:
    "What is the average age?"        -> csv_analyst
    "Who has the highest salary?"     -> csv_analyst
    "Give me a summary of the CSV."   -> csv_report_agent
    "Create a report with a chart."   -> csv_report_agent
    """,
    sub_agents=[csv_analyst, csv_report_agent],
)
```

**How it works:** the manager reads your question and transfers it to the best sub-agent. That sub-agent answers you directly.

## Step 8: Run and test the full system

```bash
adk web
```

Try these in order:

| Question | Who should answer |
|----------|-------------------|
| How many students are there? | csv_analyst |
| Which student has the highest salary? | csv_analyst |
| Give me a summary of the CSV with a chart | csv_report_agent |

In the left panel of `adk web` you can open the **Trace/Events** tab to see which agent was called and which tool it used. This is the best part to show in a workshop.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Cannot connect to Ollama | Check that Ollama is running: `ollama list` |
| Model not found | Run `ollama pull gemma4:e4b` again |
| Agent does not use the tool | Keep `temperature=0` and keep "ALWAYS use the read_csv tool" in the instruction |
| Wrong agent is picked | Make the `description` of each sub-agent clearer |
| `csv_agent` not visible in `adk web` | Check `__init__.py` and run `adk web` from the parent folder |

## Practice ideas

1. Add your own CSV with new columns and ask new questions.
2. Add a third agent that cleans missing values.
3. Change the model to a bigger Gemma and compare answers.

## Learn more

- ADK + Ollama: https://google.github.io/adk-docs/agents/models/ollama
- ADK + Gemma: https://adk.dev/agents/models/google-gemma
- Official Google codelab (Gemini + Gemma multi-agent on Cloud Run): https://codelabs.developers.google.com/next26/multi-agent-system

---

*Created by Geeta Kakrani, Google Developer Expert (AI/ML).*

# Module 2 — Phase 1 Mini-Project: Study Guide Agent

A hands-on implementation of a three-task sequential "Study Guide Agent" built twice using two different orchestration frameworks: **LangGraph** and **CrewAI**.

Given any technical topic (e.g., *Model Context Protocol*), the agent systematically produces:
1. A short plain-language explanation.
2. A practical example and one common misconception.
3. A three-question mini-quiz with an answer key.

---

## Setup & Installation

1. Ensure you have **Python 3.11+** installed in your environment.
2. Install the required dependencies:
   ```bash
   pip install openai langgraph crewai langchain-openai

```

3. Set your OpenRouter API key as an environment variable:
* **On macOS/Linux:**
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"

```

* **On Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"

```

---

## How to Run

### 1. LangGraph Implementation

Executes the explicit graph-based workflow (`START` → `explain_topic` → `create_example` → `create_quiz` → `END`).

```bash
python3 study_guide_langgraph.py

```

### 2. CrewAI Implementation

Executes the sequential multi-task workflow using a single `Study Guide Agent` and `Process.sequential`.

```bash
python3 study_guide_crewai.py

```

---

## Tested Topics

* `Model Context Protocol`
* `temperature in language models`

---

## Building the Same System Twice

### 1. What did LangGraph make explicit?

LangGraph made the entire control flow and state schema fully transparent and programmatic. By explicitly defining the typed state dictionary (`PipelineContext`) and wiring each node (`step_one` → `step_two` → `step_three`) using directed edges (`add_edge`), the data movement, state transitions, and step sequencing were completely visible and controllable at code level.

### 2. What did CrewAI automate or hide?

CrewAI abstracted away manual state management and routing boilerplate. By leveraging `Process.sequential` and simply injecting preceding tasks into the `context=[]` parameter of subsequent tasks, the framework automatically handled prompt chaining and context hand-off behind the scenes without requiring custom data-mapping logic.

### 3. What would you choose for this three-task pipeline, and why?

For a linear, straightforward three-task assembly pipeline like this, **CrewAI** is preferred for rapid prototyping due to its minimal boilerplate and natural task definition syntax. However, if the system required complex conditional branching, cyclic loops (e.g., re-running a step if quality validation fails), or deterministic state checks, **LangGraph** would be the superior and more robust architectural choice.

```

```
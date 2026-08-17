from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from dotenv import find_dotenv, load_dotenv  # noqa: E402

try:
    from crewai.events.listeners.tracing.utils import mark_first_execution_done
    mark_first_execution_done()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search_tools import web_search  # noqa: E402

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found.\n"
        "Create a file named .env in the project folder containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
        "then run this script again."
    )

# TODO 1 - build the model client (Customized initialization block)
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0.1,
    api_key=api_key,
)


# TODO 2 - wrap web_search with @tool (Custom decorated function)
@tool("Web Search Module")
def search_web(query: str) -> str:
    """Execute a live web search using the internal search utility.
    Returns markdown-formatted findings with active URLs or specific failure flags 
    such as NO_RESULTS or SEARCH_UNAVAILABLE."""
    return web_search(query)


def run_crew(question: str):
    # TODO 3 - the researcher (holds the tool)
    researcher = Agent(
        role="Lead Information Retrieval Specialist",
        goal="Perform targeted web queries to extract factual data, raw snippets, and direct source URLs.",
        backstory=(
            "You are a meticulous data hunter. You rely entirely on your search tool to gather "
            "verifiable information. You never guess or fabricate details, and if a search yields "
            "no matching records, you explicitly report 'NO_RESULTS'."
        ),
        tools=[search_web],
        llm=llm,
        max_iter=3,
        verbose=True
    )

    # TODO 4 - the writer (holds no tools)
    writer = Agent(
        role="Editorial Fact-Checking Desk Editor",
        goal="Review raw research findings, evaluate evidence quality, and draft verified summaries or gap reports.",
        backstory=(
            "You are a strict editorial controller. You inspect the research findings carefully. "
            "If verifiable evidence and valid links exist, you synthesize a concise, source-backed answer. "
            "If evidence is lacking or returns 'NO_RESULTS', you refuse to answer, detail the missing gaps, "
            "and suggest a follow-up query starting with 'NEXT SEARCH:'."
        ),
        tools=[],
        llm=llm,
        verbose=True
    )

    # TODO 5 - the research task (no agent= parameter specified in hierarchical process)
    research_task = Task(
        description=(
            f"Investigate the following inquiry using the web search tool: '{question}'. "
            "Collect precise facts, data points, and reference links."
        ),
        expected_output="A structured compilation of search findings complete with source URLs, or an explicit NO_RESULTS tag.",
    )

    # TODO 6 - the writing task (no agent= parameter)
    writing_task = Task(
        description=(
            f"Evaluate the research output for the target inquiry: '{question}'. "
            "Formulate an accurate final response incorporating source citations if verified. "
            "In cases where data is absent or returns NO_RESULTS, suppress guessing, output an honest gap analysis, "
            "and append a distinct line structured as 'NEXT SEARCH: <suggested query>'."
        ),
        expected_output="A verified factual write-up with URLs or a structured information gap report.",
    )

    # TODO 7 - Process.hierarchical + manager_llm
    # Note: manager_agent must NOT be included in the 'agents' list.
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True
    )

    result = crew.kickoff()
    return result


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or \
        "Anthropic was founded by former OpenAI employees"
    
    print("=" * 70)
    print(f"TARGET INQUIRY : {question}")
    print("=" * 70)
    
    result = run_crew(question)
    
    print("\n" + "=" * 70)
    print("CREWAI EXECUTION RESULT:")
    print("=" * 70)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
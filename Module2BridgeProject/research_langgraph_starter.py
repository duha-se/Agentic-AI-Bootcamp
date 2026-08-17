from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search_tools import NO_RESULTS, SEARCH_UNAVAILABLE, web_search  # noqa: E402

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")
base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found.\n"
        "Create a file named .env in the project folder containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
        "then run this script again."
    )


# TODO 1 - build the model client, same as every week.
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url=base_url,
    api_key=api_key,
)


# ---------------------------------------------------------------------------
# STATE - the luggage that travels the route.
# ---------------------------------------------------------------------------
class DeskState(TypedDict):
    question: str      # the claim or topic that came in
    query: str         # plan_query writes here
    evidence: str      # run_search writes here
    verdict: str       # assess writes here  <- THIS DRIVES THE ROUTE
    reasoning: str     # assess writes here
    output: str        # write_answer OR report_gap writes here
    route_taken: str   # so you can PROVE which path ran


def plan_query(state: DeskState) -> DeskState:
    """Node 1: turn the question into something worth searching for."""
    prompt = (
        "Extract only the 2 to 4 most important core keywords or entities from this question for a web search. "
        "Do NOT write a full sentence. Output ONLY the short keywords and nothing else.\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt)
    state["query"] = response.content.strip()
    return state


def run_search(state: DeskState) -> DeskState:
    """Node 2: the tool. There is no model call in this node at all."""
    # TODO 3 - call web_search(state["query"]), store the result in
    #          state["evidence"], and RETURN state.
    state["evidence"] = web_search(state["query"])
    return state


def assess(state: DeskState) -> DeskState:
    """Node 3: is this evidence actually good enough to answer with?"""
    # TODO 4 - handle the two failure markers BEFORE you call the model.
    evidence = state["evidence"]
    
    if evidence == SEARCH_UNAVAILABLE:
        state["verdict"] = "NOT_ENOUGH"
        state["reasoning"] = "Search infrastructure is unavailable (network/service error)."
        return state

    if evidence == NO_RESULTS:
        state["verdict"] = "NOT_ENOUGH"
        state["reasoning"] = "Search ran successfully, but genuinely found nothing useful for this query."
        return state

    # TODO 5 - now ask the model whether the evidence answers the question.
    prompt = f"""
    Question/Claim: {state['question']}
    Evidence retrieved:
    {evidence}

    Does this evidence contain enough factual information to answer the question honestly and accurately?
    Provide 1-3 short bullet points of reasoning, and on the FINAL line output EXACTLY one word: either ENOUGH or NOT_ENOUGH.
    """
    response = llm.invoke(prompt)
    full_text = response.content.strip()
    state["reasoning"] = full_text
    state["verdict"] = read_verdict(full_text)
    return state


def read_verdict(raw: str) -> str:
    """Pull the verdict out of the model's free text.

    TODO 6 - return "ENOUGH" or "NOT_ENOUGH".
    """
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if not lines:
        return "NOT_ENOUGH"
    
    last_line = lines[-1].upper()
    
    # Safe direction & checking the "NOT_ENOUGH" vs "ENOUGH" containment trap
    if "NOT_ENOUGH" in last_line:
        return "NOT_ENOUGH"
    elif "ENOUGH" in last_line:
        return "ENOUGH"
        
    return "NOT_ENOUGH"


def choose_next(state: DeskState) -> Literal["write_answer", "report_gap"]:
    """THE ROUTER. Runs no model, writes no state. It only names the next node."""
    # TODO 7 - return "write_answer" if the verdict is ENOUGH, else "report_gap".
    if state["verdict"] == "ENOUGH":
        return "write_answer"
    return "report_gap"


def write_answer(state: DeskState) -> DeskState:
    """Node 4a: the ENOUGH path."""
    # TODO 8 - answer the question using ONLY state["evidence"].
    prompt = f"""
    Answer the question using ONLY the provided evidence.
    Rules:
    - Every claim must come from the evidence, invent nothing.
    - Quote at least one source URL from the evidence.
    - If part of the question is not covered, say so.
    - Under 180 words.

    Question: {state['question']}
    Evidence: {state['evidence']}
    """
    response = llm.invoke(prompt)
    state["output"] = response.content.strip()
    state["route_taken"] = "answered"
    return state


def report_gap(state: DeskState) -> DeskState:
    """Node 4b: the NOT_ENOUGH path. This node must NOT answer the question."""
    # TODO 9 - report the gap instead of answering.
    prompt = f"""
    The claim or question could not be verified with the search results.
    Provide:
    - One sentence saying plainly this could not be verified.
    - One or two bullets on what is missing.
    - A final line in exactly this form: NEXT SEARCH: <the one query you would run next>

    Question: {state['question']}
    Evidence status: {state['evidence']}
    """
    response = llm.invoke(prompt)
    state["output"] = response.content.strip()
    state["route_taken"] = "gap reported"
    return state


def build_graph():
    graph = StateGraph(DeskState)

    # TODO 10 - register all five nodes.
    graph.add_node("plan_query", plan_query)
    graph.add_node("run_search", run_search)
    graph.add_node("assess", assess)
    graph.add_node("write_answer", write_answer)
    graph.add_node("report_gap", report_gap)

    # TODO 11 - the fixed part of the route:
    graph.set_entry_point("plan_query")
    graph.add_edge("plan_query", "run_search")
    graph.add_edge("run_search", "assess")

    # TODO 12 - THE LINE (Conditional Edges).
    graph.add_conditional_edges(
        "assess",
        choose_next,
        {
            "write_answer": "write_answer",
            "report_gap": "report_gap",
        },
    )

    # TODO 13 - send BOTH write_answer and report_gap to END.
    graph.add_edge("write_answer", END)
    graph.add_edge("report_gap", END)

    return graph.compile()


def run(question: str) -> DeskState:
    return build_graph().invoke({
        "question": question,
        "query": "",
        "evidence": "",
        "verdict": "",
        "reasoning": "",
        "output": "",
        "route_taken": "",
    })


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or \
        "Anthropic was founded by former OpenAI employees"

    result = run(question)

    print("=" * 70)
    print(f"QUESTION : {result['question']}")
    print("=" * 70)
    print(f"\n[1] SEARCH QUERY\n{result['query']}")
    print(f"\n[2] EVIDENCE ({len(result['evidence'])} chars)")
    print(result["evidence"][:700])
    print(f"\n[3] ASSESSMENT\n{result['reasoning']}")
    print("\n" + "=" * 70)
    print(f"VERDICT : {result['verdict']}")
    print(f"ROUTE   : {result['route_taken'].upper()}")
    print("=" * 70)
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
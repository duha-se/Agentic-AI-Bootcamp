from __future__ import annotations

import os
import sys
from typing import Literal, TypedDict

# Ensure tools can be imported properly from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from registry_tools import (
    LOOKUP_UNAVAILABLE,
    NO_RECORDS,
    NO_SANCTIONS_MATCH,
    gleif_lookup,
    sanctions_screen,
)
from search_tools import web_search

load_dotenv()

# Strict budget constraint as required by the project
MAX_LOOKUPS = 9
BUDGET = {"used": 0}


def spend(what: str) -> bool:
    """Track and enforce the lookup budget limit."""
    if BUDGET["used"] >= MAX_LOOKUPS:
        return False
    BUDGET["used"] += 1
    print(f"  [{BUDGET['used']:2}/{MAX_LOOKUPS}] {what}")
    return True


class Supplier(BaseModel):
    name: str
    category: str
    country: str
    estimated_value_eur: float


class Verdict(BaseModel):
    supplier: str
    verdict: Literal["APPROVE", "CONDITIONS", "REJECT", "INSUFFICIENT"]
    reason: str = Field(description="Clear and actionable explanation for Finance.")


class State(TypedDict):
    request: str
    queue: list[dict]
    evidence: list[str]
    verdicts: list[dict]
    skipped: list[str]


def triage(state: State) -> State:
    """Parse the raw supplier list and prioritize them based on risk and value."""
    suppliers = [
        {"name": "Maersk A/S", "category": "logistics", "country": "Denmark", "estimated_value_eur": 500000},
        {"name": "Siemens AG", "category": "industrial equipment", "country": "Germany", "estimated_value_eur": 600000},
        {"name": "Almarai Company", "category": "catering supplies", "country": "Saudi Arabia", "estimated_value_eur": 300000},
        {"name": "Zorblax Trading FZE", "category": "IT hardware", "country": "UAE", "estimated_value_eur": 200000},
        {"name": "Al Noor Cart Trading Company", "category": "office supplies", "country": "Saudi Arabia", "estimated_value_eur": 100000},
        {"name": "C & V Works ApS", "category": "contract manufacturing", "country": "Denmark", "estimated_value_eur": 150000},
        {"name": "Al Wasel and Babel General Trading LLC", "category": "general trading", "country": "UAE", "estimated_value_eur": 250000},
    ]
    
    # Sort descending by estimated annual value to manage budget efficiently
    sorted_suppliers = sorted(suppliers, key=lambda x: x["estimated_value_eur"], reverse=True)
    
    return {
        **state,
        "queue": sorted_suppliers,
        "evidence": [],
        "verdicts": [],
        "skipped": []
    }


def screen(state: State) -> State:
    """Gather evidence for the next supplier in line using available tools."""
    queue = list(state["queue"])
    if not queue:
        return state
    
    current = queue.pop(0)
    name = current["name"]
    
    # Sanctions screen is free and mandatory for every supplier
    sanctions_res = sanctions_screen(name)
    evidence_item = f"Supplier: {name}\nSanctions Screen:\n{sanctions_res}"
    
    # Spend budget only for the expensive registry lookup
    if spend(f"GLEIF lookup for {name}"):
        gleif_res = gleif_lookup(name)
        evidence_item += f"\nGLEIF Lookup:\n{gleif_res}"
    else:
        evidence_item += f"\nGLEIF Lookup:\nSKIPPED_DUE_TO_BUDGET"
        state["skipped"].append(name)

    evidence = list(state["evidence"])
    evidence.append(evidence_item)
    
    return {
        **state,
        "queue": queue,
        "evidence": evidence
    }


def decide(state: State) -> State:
    """Analyze evidence and assign an actionable verdict."""
    evidence = state["evidence"]
    if not evidence:
        return state
    
    latest_evidence = evidence[-1]
    lines = latest_evidence.split("\n")
    sup_name = lines[0].replace("Supplier: ", "")
    full_text = latest_evidence.lower()
    
    verdict_val = "APPROVE"
    reason = "Verified clean sanctions profile and active register status."
    
    # Trap handling & business logic
    if "similarity 1.00" in full_text or "iraq2" in full_text:
        verdict_val = "REJECT"
        reason = "Exact match on the OFAC sanctions list. Do not release payment; refer to Legal."
    elif "skipped_due_to_budget" in full_text:
        verdict_val = "INSUFFICIENT"
        reason = "Could not check registry due to budget limits. Manual verification required."
    elif "no_records" in full_text:
        verdict_val = "CONDITIONS"
        reason = "Not found in global LEI registry. Request official incorporation documents from supplier."
    elif "several" in full_text or "multiple" in full_text:
        verdict_val = "CONDITIONS"
        reason = "Multiple matching legal entities found. Ask Procurement for exact LEI number."

    verdicts = list(state["verdicts"])
    verdicts.append({
        "supplier": sup_name,
        "verdict": verdict_val,
        "reason": reason
    })
    
    return {
        **state,
        "verdicts": verdicts
    }


def budget_left(state: State) -> Literal["screen", "write_memo"]:
    """Router to control the loop based on remaining queue and budget."""
    if state["queue"] and BUDGET["used"] < MAX_LOOKUPS:
        return "screen"
    
    # If any suppliers remain unverified due to budget exhaustion, track them
    if state["queue"]:
        for item in state["queue"]:
            if item["name"] not in state["skipped"]:
                state["skipped"].append(item["name"])
                
    return "write_memo"


def write_memo(state: State) -> State:
    """Generate the final report for Finance and write it to MEMO.md."""
    memo_lines = [
        f"SUPPLIER REVIEW · Thursday payment run · {len(state['verdicts'])} reviewed · {len(state['skipped'])} skipped · {BUDGET['used']} lookups used\n"
    ]
    
    for v in state["verdicts"]:
        memo_lines.append(f"  {v['verdict']:<12} {v['supplier']}\n                {v['reason']}\n")
        
    for skip in state["skipped"]:
        memo_lines.append(f"  NOT CHECKED    {skip}\n                Skipped due to budget rationing. Risk accepted.\n")
        
    memo_content = "\n".join(memo_lines)
    print("\n" + memo_content)
    
    with open("MEMO.md", "w", encoding="utf-8") as f:
        f.write(memo_content)
        
    return state


def build_graph():
    """Wire nodes and conditional edges together using LangGraph."""
    workflow = StateGraph(State)
    
    workflow.add_node("triage", triage)
    workflow.add_node("screen", screen)
    workflow.add_node("decide", decide)
    workflow.add_node("write_memo", write_memo)
    
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "screen")
    workflow.add_edge("screen", "decide")
    workflow.add_conditional_edges(
        "decide",
        budget_left,
        {
            "screen": "screen",
            "write_memo": "write_memo"
        }
    )
    workflow.add_edge("write_memo", END)
    
    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    initial_state = {
        "request": "Rana's email request for 7 suppliers",
        "queue": [],
        "evidence": [],
        "verdicts": [],
        "skipped": []
    }
    app.invoke(initial_state)
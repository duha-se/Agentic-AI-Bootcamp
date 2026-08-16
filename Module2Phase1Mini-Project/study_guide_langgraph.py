import os
from typing import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, START, END


MODEL_ENGINE = "openai/gpt-4o-mini"
API_ENDPOINT = "https://openrouter.ai/api/v1"

ai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url=API_ENDPOINT
)

def query_llm(instruction: str) -> str:
    response = ai_client.chat.completions.create(
        model=MODEL_ENGINE,
        messages=[{"role": "user", "content": instruction}],
        temperature=0.2
    )
    return response.choices[0].message.content

class PipelineContext(TypedDict):
    subject: str
    summary_text: str
    practical_case: str
    final_assessment: str

# Node 1: generate a concise overview of the subject
def fn_step_one(ctx: PipelineContext) -> dict:
    prompt = f"Provide a brief, clear 2-3 sentence overview of '{ctx['subject']}'. No stats, no quiz."
    return {"summary_text": query_llm(prompt)}

# Node 2: generate a practical case and a common misconception
def fn_step_two(ctx: PipelineContext) -> dict:
    prompt = f"Subject: {ctx['subject']}\nOverview: {ctx['summary_text']}\nGive one real-world example and one common misconception, clearly distinguishing them."
    return {"practical_case": query_llm(prompt)}

# Node 3: joining everything into a structured guide
def fn_step_three(ctx: PipelineContext) -> dict:
    prompt = f"""
    Subject: {ctx['subject']}
    Overview: {ctx['summary_text']}
    Case & Misconception: {ctx['practical_case']}
    
    Compile everything into a structured guide with these exact sections:
    ## Explanation
    ## Example and misconception
    ## Quiz (3 questions with an answer key)
    """
    return {"final_assessment": query_llm(prompt)}

# Building the graph and defining edges
builder = StateGraph(PipelineContext)
builder.add_node("step_one", fn_step_one)
builder.add_node("step_two", fn_step_two)
builder.add_node("step_three", fn_step_three)

builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", "step_three")
builder.add_edge("step_three", END)

compiled_graph = builder.compile()

if __name__ == "__main__":
    target_topic = "Model Context Protocol"
    print(f"=== LangGraph Execution for: {target_topic} ===\n")
    output_state = compiled_graph.invoke({"subject": target_topic})
    print(output_state["final_assessment"])
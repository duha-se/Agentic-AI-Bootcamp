import os
from crewai import Agent, Task, Crew, Process

# Setting up environment variables for OpenAI API access
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

MODEL_ENGINE = "openai/gpt-4o-mini"

# Agent configuration for the academic mentor
mentor_agent = Agent(
    role="Master Academic Tutor",
    goal="Synthesize precise, verified, and well-structured educational study packs.",
    backstory="An expert curriculum designer specialized in translating complex technical specifications into digestible study materials.",
    llm=MODEL_ENGINE, # Passing the model name as a string directly
    verbose=False
)

def execute_study_crew(topic_name: str):
    # Task 1: Generate a simple explanation
    task_intro = Task(
        description=f"Write a 2-3 sentence clear explanation for '{topic_name}'. Do not include a quiz or fake statistics.",
        expected_output="A concise plain-language definition.",
        agent=mentor_agent
    )

    # Task 2: Generate a practical case and a common misconception
    task_application = Task(
        description=f"Using the previous definition of '{topic_name}', generate a concrete practical use-case and a separate common misconception.",
        expected_output="A breakdown containing an explicit example and a misconception.",
        agent=mentor_agent,
        context=[task_intro]
    )

    # Task 3: Compile everything into a structured study guide with a quiz
    task_evaluation = Task(
        description=f"Synthesize the final guide for '{topic_name}' incorporating all previous sections under these headers:\n## Explanation\n## Example and misconception\n## Quiz (3 questions with keys).",
        expected_output="A fully compiled study guide with explanation, case/misconception, and quiz.",
        agent=mentor_agent,
        context=[task_intro, task_application]
    )

    academic_crew = Crew(
        agents=[mentor_agent],
        tasks=[task_intro, task_application, task_evaluation],
        process=Process.sequential,
        tracing=False
    )

    return academic_crew.kickoff(inputs={"topic": topic_name})

if __name__ == "__main__":
    target_topic = "Model Context Protocol"
    print(f"=== CrewAI Execution for: {target_topic} ===\n")
    final_result = execute_study_crew(target_topic)
    print(final_result)
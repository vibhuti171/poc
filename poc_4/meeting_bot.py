from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
import os

# --- Load your .env file ---
load_dotenv()

# --- Initialize LLM with correct Groq model name ---
llm = LLM(
    model="groq/llama-3.3-70b-versatile",  # Correct Groq model name with prefix
    temperature=0.7
)

# --- Sample meeting notes ---
meeting_transcript = """
Hey, let's finish the homepage redesign by next Wednesday. 
Also, Satyam will handle the pitch deck, and Vibhuti will fix backend bugs by Friday. 
Don't forget to send the proposal to the investor. 
"""

# --- Agent 1: Summarizer ---
summarizer_agent = Agent(
    role="Meeting Summarizer",
    goal="Generate a short and clear summary of the key points discussed in the meeting",
    backstory="You are great at distilling conversations into professional summaries.",
    verbose=True,
    llm=llm
)

# --- Agent 2: Task Extractor ---
task_extractor_agent = Agent(
    role="Task Extractor",
    goal="Extract all tasks, deadlines, and people responsible from the meeting summary",
    backstory="You are a project manager bot who identifies action items and organizes them clearly.",
    verbose=True,
    llm=llm
)

# --- Task 1: Summarization ---
summary_task = Task(
    description=f"Summarize the following meeting transcript:\n\n{meeting_transcript}",
    expected_output="A short summary (2-3 sentences) of the key points.",
    agent=summarizer_agent
)

# --- Task 2: Task Extraction ---
task_extraction_task = Task(
    description="Based on the meeting summary, extract all action items with task descriptions, due dates, and responsible persons in JSON format.",
    expected_output="""
A JSON list like:
[
  {"task": "...", "due": "...", "assigned_to": "..."},
  ...
]
""",
    agent=task_extractor_agent
)

# --- Create and run the Crew ---
crew = Crew(
    agents=[summarizer_agent, task_extractor_agent],
    tasks=[summary_task, task_extraction_task],
    verbose=True
)

result = crew.kickoff()
print("\n📌 FINAL RESULT:\n", result)
import os

from dotenv import load_dotenv

from taskflow.llm import GeminiClient
from taskflow.flow import Task, TaskFlow
from taskflow.agents import Commiter, Evaluator, Reviewer
from taskflow.tools import diff_tool
from taskflow.mock import create_temp_git_repo


if __name__ == "__main__":
    load_dotenv()

    print("Initializing LegionAI system...")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-preview-05-20") # Using a more recent model
    gemini_client = GeminiClient(model_name=DEFAULT_MODEL)

    project_dir = os.path.abspath(os.path.join(os.getcwd(), "tmp_test_project")) # Use a local tmp dir
    create_temp_git_repo(project_dir)

    task = Task(
        prompt=f"""
        Generate a concise commit message for the staged changes in the project '{project_dir}'.
        """,
        needs_approval=True
    )

    review_task = Task(
        prompt=f"""
        Generate a concise REVIEW about changes in the project '{project_dir}'.
        """,
        needs_approval=False
    )

    # Initialize agents
    commiter_agent = Commiter(
        model=gemini_client,
        system_prompt="""
You are a senior programmer that explains hard concepts clearly and are very succinctly in your messages. You can evaluate changes in a project just by reading the diff output from git.

You MUST use the `diff_tool` to get the changes in the project.

Respond ONLY in the JSON format (example):

{"message": "Refactor GitReviewer for improved LLM integration and REPL functionality", "details": ["Introduced a `_get_config` method in `LLMGoogle` to centralize LLM calls.", "Refactored `main.py` to use a new `init_repl` function, streamlining the application's entry point and focusing on a REPL interface.", "Moved the `Message` Pydantic model to a dedicated `models.py`"]}
""",
        available_tools={'diff_tool': diff_tool} # Add diff_tool to the Reviewer's tools
    )

    evaluator_agent = Evaluator(
        model=gemini_client,
        system_prompt="""
You are a senior programmer that has attention to details and likes very clear texts. You made code reviews and evaluate the quality of the commit messages based on the diff changes.
You MUST use the `diff_tool` to get the changes in the project.
If your evaluation is positive, just respond with 'Commit message accepted', but
if the commit message has any problems respond with 'Bad commit message', two new lines and the motive.
""",
        available_tools={'diff_tool': diff_tool} # Add diff_tool to the Reviewer's tools
    )

    reviewer_agent = Reviewer( # Initialize the Reviewer agent here
        model=gemini_client,
        system_prompt="""
You are a meticulous code reviewer. Your task is to provide a concise and constructive review of the given code changes, focusing on clarity, potential issues, and adherence to best practices. Summarize the key changes and any recommendations.
You MUST use the `diff_tool` to get the staged changes in the project.
""",
        available_tools={'diff_tool': diff_tool} # Add diff_tool to the Reviewer's tools
    )

    flow = TaskFlow(model=gemini_client) # LegionAI uses its own LLM instance for orchestration
    flow.add(commiter_agent)
    flow.add(evaluator_agent)
    flow.add(reviewer_agent)

    flow.run(task, max_attempts=3)

    print("\n--- Task execution finished. Memory content: ---")
    for interaction in flow.memory.get_history():
        print(f"[{interaction['role'].upper()}]: {interaction['content']}")

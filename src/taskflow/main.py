import os

import git
from dotenv import load_dotenv

from taskflow.llm import GeminiClient
from taskflow.flow import Task, TaskFlow
from taskflow.agents import Commiter, Evaluator, Reviewer
from taskflow.tools import diff_tool


if __name__ == "__main__":
    print("Initializing LegionAI system...")
    # It's good practice to get the model name from an environment variable or config
    # For testing, you might default it if the env var isn't set.
    load_dotenv()
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-preview-05-20") # Using a more recent model
    gemini_client = GeminiClient(model_name=DEFAULT_MODEL)


    # Define a valid project directory for testing.
    # This directory should ideally be a git repository with some staged changes for diff_tool to work.
    # For example:
    # 1. Create a directory: `mkdir /tmp/my_test_project`
    # 2. Navigate into it: `cd /tmp/my_test_project`
    # 3. Initialize git: `git init`
    # 4. Create a file: `echo "Hello World" > test.txt`
    # 5. Add to stage: `git add test.txt`
    # You will then see the diff.
    project_dir = os.path.abspath(os.path.join(os.getcwd(), "tmp_test_project")) # Use a local tmp dir

    # Create the temporary project directory if it doesn't exist
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        print(f"Created temporary project directory: {project_dir}")

    # Initialize a dummy git repo for testing diff_tool
    try:
        repo = git.Repo(project_dir)
        print(f"Found existing Git repo at {project_dir}")
    except git.InvalidGitRepositoryError:
        repo = git.Repo.init(project_dir)
        print(f"Initialized new Git repo at {project_dir}")

    # Create a dummy file with changes and stage it
    dummy_file_path = os.path.join(project_dir, "dummy_file.txt")
    with open(dummy_file_path, "w") as f:
        f.write("This is the first line.\n")
        f.write("This is the second line, with a change.\n")

    # Add some more content to show a diff
    with open(dummy_file_path, "a") as f:
        f.write("A new line added.\n")

    try:
        repo.index.add([dummy_file_path])
        print(f"Staged changes in {dummy_file_path} for diff_tool testing.")
    except Exception as e:
        print(f"Could not stage changes in {dummy_file_path}: {e}. Diff tool might not show output.")
        # Attempt to make a commit so future diffs can be generated relative to it
        if not repo.heads:
            try:
                repo.index.commit("Initial commit for testing")
                print("Made an initial commit.")
                # Now add new changes to show a diff
                with open(dummy_file_path, "a") as f:
                    f.write("Another line after initial commit.\n")
                repo.index.add([dummy_file_path])
                print("Staged new changes after initial commit.")
            except Exception as commit_e:
                print(f"Could not make initial commit or stage new changes: {commit_e}. Diff tool might not work as expected.")


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

    # Initialize LegionAI
    flow = TaskFlow(model=gemini_client) # LegionAI uses its own LLM instance for orchestration
    flow.add(commiter_agent)
    flow.add(evaluator_agent)
    flow.add(reviewer_agent)


    # Run the task
    flow.run(task, max_attempts=3)

    print("\n--- Task execution finished. Memory content: ---")
    for interaction in flow.memory.get_history():
        print(f"[{interaction['role'].upper()}]: {interaction['content']}")

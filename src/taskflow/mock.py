import os

import git

def create_temp_git_repo(project_dir: str):

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
                with open(dummy_file_path, "a") as f:
                    f.write("Another line after initial commit.\n")
                repo.index.add([dummy_file_path])
                print("Staged new changes after initial commit.")
            except Exception as commit_e:
                print(f"Could not make initial commit or stage new changes: {commit_e}. Diff tool might not work as expected.")
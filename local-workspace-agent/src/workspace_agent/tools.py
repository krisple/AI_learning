import subprocess

from langchain.tools import tool
from langgraph.runtime import get_runtime

from workspace_agent.context import RuntimeContext


@tool
def git_status() -> str:
    """Return the git status of the current workspace repository."""
    try:
        runtime = get_runtime(RuntimeContext)
        workspace_path = runtime.context["workspace_path"]

        result = subprocess.run(
            ["git", "status"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout

    except subprocess.CalledProcessError as e:
        return f"Error running git status: {e.stderr}"

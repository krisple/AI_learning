from langchain.tools import tool

from workspace_agent.middleware import AgentModeMiddleware
from workspace_agent.tools import git_status


@tool
def write_file() -> str:
    """Example tool that may modify a file."""
    return "written"


@tool
def read_file() -> str:
    """Example tool that only reads a file."""
    return "contents"


read_file.metadata = {"readOnlyHint": True}


def test_git_status_is_allowed_in_read_mode() -> None:
    assert AgentModeMiddleware._is_read_only(git_status)


def test_mcp_annotation_marks_tool_as_read_only() -> None:
    assert AgentModeMiddleware._is_read_only(read_file)


def test_unannotated_tool_is_not_read_only() -> None:
    assert not AgentModeMiddleware._is_read_only(write_file)

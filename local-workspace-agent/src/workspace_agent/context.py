from typing import TypedDict

from workspace_agent.mode import AgentMode


class RuntimeContext(TypedDict):
    """Values available to tools and middleware during an agent run."""

    workspace_path: str
    mode: AgentMode

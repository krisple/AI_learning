from enum import StrEnum

class AgentMode(StrEnum):
    """Controls whether the agent may use tools that modify the workspace."""

    READ = "read"
    WRITE = "write"

from typing import Any

from langchain.agents.middleware import (
    AgentState,
    HumanInTheLoopMiddleware,
)

from workspace_agent.context import RuntimeContext

write_approval_middleware = HumanInTheLoopMiddleware[
    AgentState[Any],
    RuntimeContext,
    Any,
](
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]},
        "edit_file": {"allowed_decisions": ["approve", "reject"]},
        "move_file": {"allowed_decisions": ["approve", "reject"]},
        "create_directory": {"allowed_decisions": ["approve", "reject"]},
    }
)

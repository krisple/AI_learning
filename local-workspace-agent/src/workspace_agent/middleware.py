from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import BaseTool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from workspace_agent.context import RuntimeContext
from workspace_agent.mode import AgentMode

LOCAL_READ_ONLY_TOOLS = {"git_status"}


class AgentModeMiddleware(AgentMiddleware[AgentState[Any], RuntimeContext]):
    """Expose only tools allowed by the agent's current mode."""

    def __init__(self) -> None:
        self._mcp_tools: list[BaseTool] = []

    def set_mcp_tools(self, tools: list[BaseTool]) -> None:
        self._mcp_tools = tools

    def clear_mcp_tools(self) -> None:
        self._mcp_tools = []

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[RuntimeContext],
        handler: Callable[
            [ModelRequest[RuntimeContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any] | AIMessage:

        tools = [*request.tools, *self._mcp_tools]

        if request.runtime.context["mode"] == AgentMode.READ:
            tools = [tool for tool in tools if self._is_read_only(tool)]

        return await handler(request.override(tools=tools))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:

        for tool in self._mcp_tools:
            if tool.name == request.tool_call["name"]:
                return await handler(request.override(tool=tool))

        return await handler(request)

    @staticmethod
    def _is_read_only(tool: BaseTool | dict[str, Any]) -> bool:

        if isinstance(tool, dict):
            return False

        metadata = tool.metadata or {}
        return (
            tool.name in LOCAL_READ_ONLY_TOOLS or metadata.get("readOnlyHint") is True
        )

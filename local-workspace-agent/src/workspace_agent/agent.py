from typing import Any

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from workspace_agent.context import RuntimeContext
from workspace_agent.filesystem_tools import get_filesystem_tools
from workspace_agent.middleware import AgentModeMiddleware
from workspace_agent.mode import AgentMode
from workspace_agent.tools import git_status


class WorkspaceAgent:
    """Run a LangChain agent against a local workspace."""

    def __init__(
        self, model_name: str, workspace_path: str, agent_mode: AgentMode
    ) -> None:

        self._context = RuntimeContext(
            workspace_path=workspace_path,
            mode=agent_mode,
        )

        self._middleware = AgentModeMiddleware()

        self._agent = create_agent(
            model=self._create_model(model_name),
            tools=[git_status],
            context_schema=RuntimeContext,
            middleware=[self._middleware],
        )

    async def run_request(self, question: str) -> dict[str, Any]:
        """Send one user request to the agent and return its full state."""

        async with get_filesystem_tools(
            self._context["workspace_path"]
        ) as filesystem_tools:
            self._middleware.set_mcp_tools(filesystem_tools)

            try:
                return await self._agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question,
                            }
                        ]
                    },
                    context=self._context,
                )
            finally:
                self._middleware.clear_mcp_tools()

    @staticmethod
    def _create_model(model_name: str) -> ChatOllama:
        return ChatOllama(
            model=model_name,
            temperature=1,
            reasoning=False,
        )

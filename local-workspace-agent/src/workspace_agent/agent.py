from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import InputAgentState
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from workspace_agent.context import RuntimeContext
from workspace_agent.filesystem_tools import get_filesystem_tools
from workspace_agent.middleware import AgentModeMiddleware
from workspace_agent.mode import AgentMode
from workspace_agent.tools import git_status
from workspace_agent.write_approval_hitl_middleware import write_approval_middleware


class WorkspaceAgent:
    """Run a LangChain agent against a local workspace."""

    def __init__(
        self,
        model_name: str,
        workspace_path: str,
        agent_mode: AgentMode,
        checkpointer: BaseCheckpointSaver[Any],
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
            middleware=[
                self._middleware,
                write_approval_middleware,
            ],
            checkpointer=checkpointer,
        )

    async def run_request(
        self,
        question: str,
        conversation_id: int,
    ) -> dict[str, Any]:
        """Send one user request to the agent and return its full state."""

        return await self._invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            conversation_id,
        )

    async def resume_request(
        self,
        decisions: dict[str, Any],
        conversation_id: int,
    ) -> dict[str, Any]:
        """Resume an interrupted request with the user's approval decisions."""

        return await self._invoke(Command(resume=decisions), conversation_id)

    async def _invoke(
        self,
        request: InputAgentState | Command[Any],
        conversation_id: int,
    ) -> dict[str, Any]:
        """Invoke the agent while its filesystem MCP session is available."""

        config = self._conversation_config(conversation_id)

        async with get_filesystem_tools(
            self._context["workspace_path"]
        ) as filesystem_tools:
            self._middleware.set_mcp_tools(filesystem_tools)

            try:
                return await self._agent.ainvoke(
                    request,
                    config=config,
                    context=self._context,
                )
            finally:
                self._middleware.clear_mcp_tools()

    async def get_conversation_messages(
        self,
        conversation_id: int,
    ) -> list[BaseMessage]:
        """Load the latest persisted messages for a conversation."""

        state = await self._agent.aget_state(self._conversation_config(conversation_id))
        return list(state.values.get("messages", []))

    @staticmethod
    def _conversation_config(conversation_id: int) -> RunnableConfig:
        return {"configurable": {"thread_id": str(conversation_id)}}

    @staticmethod
    def _create_model(model_name: str) -> ChatOllama:
        return ChatOllama(
            model=model_name,
            temperature=1,
            reasoning=False,
        )

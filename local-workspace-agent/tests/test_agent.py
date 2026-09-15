import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.messages import AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from workspace_agent import agent as agent_module
from workspace_agent.agent import WorkspaceAgent
from workspace_agent.mode import AgentMode


def test_conversation_id_is_used_as_the_checkpoint_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeGraph:
        async def astream(
            self,
            values: dict[str, Any] | Command[Any],
            *,
            config: dict[str, Any],
            context: dict[str, Any],
            stream_mode: str,
        ) -> AsyncIterator[tuple[AIMessageChunk, dict[str, Any]]]:
            captured["invoke_config"] = config
            captured.setdefault("invoke_inputs", []).append(values)
            captured["stream_mode"] = stream_mode
            yield (
                AIMessageChunk(content="ans"),
                {
                    "langgraph_node": "model",
                    "langgraph_step": 1,
                },
            )
            yield (
                AIMessageChunk(content="wer"),
                {
                    "langgraph_node": "model",
                    "langgraph_step": 1,
                },
            )

        async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
            captured["state_config"] = config
            return SimpleNamespace(
                values={"messages": [AIMessage("answer")]},
                interrupts=(),
            )

    def fake_create_agent(**kwargs: Any) -> FakeGraph:
        captured["checkpointer"] = kwargs["checkpointer"]
        return FakeGraph()

    @asynccontextmanager
    async def fake_filesystem_tools(
        workspace_path: str,
    ) -> AsyncIterator[list[BaseTool]]:
        captured["workspace_path"] = workspace_path
        yield []

    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(
        agent_module,
        "get_filesystem_tools",
        fake_filesystem_tools,
    )

    checkpointer = InMemorySaver()
    agent = WorkspaceAgent(
        model_name="test-model",
        workspace_path="/tmp/workspace",
        agent_mode=AgentMode.READ,
        checkpointer=checkpointer,
    )

    async def scenario() -> None:
        streamed_text: list[str] = []
        await agent.run_request(
            "hello",
            conversation_id=42,
            on_text_chunk=lambda text, _stream_id: streamed_text.append(text),
        )
        await agent.resume_request(
            {
                "interrupt-1": {
                    "decisions": [{"type": "approve"}],
                }
            },
            conversation_id=42,
        )
        messages = await agent.get_conversation_messages(conversation_id=42)
        assert messages == [AIMessage("answer")]
        assert streamed_text == ["ans", "wer"]

    asyncio.run(scenario())

    expected_config = {"configurable": {"thread_id": "42"}}
    assert captured["invoke_config"] == expected_config
    assert captured["state_config"] == expected_config
    assert captured["checkpointer"] is checkpointer
    assert captured["stream_mode"] == "messages"

    initial_input, resume_input = captured["invoke_inputs"]
    assert initial_input == {
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert isinstance(resume_input, Command)
    assert resume_input.resume == {"interrupt-1": {"decisions": [{"type": "approve"}]}}

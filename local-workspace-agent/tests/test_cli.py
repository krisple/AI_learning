import asyncio
from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import ToolCall
from langgraph.types import Interrupt

from workspace_agent.agent import WorkspaceAgent
from workspace_agent.cli import (
    ChatCLI,
    CommandError,
    Terminal,
    _recent_visible_messages,
    _visible_messages,
)
from workspace_agent.conversations import ConversationStore


def test_visible_messages_hide_tool_messages_and_empty_agent_calls() -> None:
    messages = [
        HumanMessage("hello"),
        AIMessage(
            content="I will inspect the file first.",
            tool_calls=[{"name": "read", "args": {}, "id": "1"}],
        ),
        ToolMessage("contents", tool_call_id="1"),
        AIMessage("welcome"),
    ]

    assert _visible_messages(messages) == [messages[0], messages[3]]


def test_recent_messages_keep_chronological_order() -> None:
    messages = [
        HumanMessage("oldest"),
        AIMessage("middle"),
        HumanMessage("newest"),
    ]

    assert _recent_visible_messages(messages, limit=2) == messages[1:]


def test_history_limit_validation() -> None:
    assert ChatCLI._optional_limit([], 10, "/history") == 10
    assert ChatCLI._optional_limit(["5"], 10, "/history") == 5

    try:
        ChatCLI._optional_limit(["0"], 10, "/history")
    except CommandError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("Expected a CommandError")


def test_terminal_distinguishes_user_and_agent(capsys) -> None:
    terminal = Terminal(use_color=False)

    terminal.show_message(HumanMessage("hello"))
    terminal.show_message(AIMessage("hi"))

    assert capsys.readouterr().out == "\nYOU\n  hello\n\nAGENT\n  hi\n"


def test_delete_active_conversation_clears_it(capsys) -> None:
    class FakeConversationStore:
        async def delete(self, conversation_id: int) -> bool:
            return conversation_id == 7

    cli = ChatCLI(
        cast(WorkspaceAgent, object()),
        cast(ConversationStore, FakeConversationStore()),
        Terminal(use_color=False),
    )
    cli._conversation_id = 7

    assert asyncio.run(cli._handle_command("/delete 7")) is True

    assert cli._conversation_id is None
    assert capsys.readouterr().out == "Deleted conversation #7.\n"


def test_write_approval_resumes_the_interrupted_conversation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}
    tool_call: ToolCall = {
        "name": "write_file",
        "args": {"path": "notes.txt", "content": "hello"},
        "id": "tool-1",
    }

    class FakeAgent:
        async def run_request(
            self,
            question: str,
            conversation_id: int,
            on_text_chunk: Callable[[str, str], None] | None = None,
        ) -> dict[str, Any]:
            return {
                "messages": [AIMessage(content="", tool_calls=[tool_call])],
                "__interrupt__": (
                    Interrupt(
                        id="interrupt-1",
                        value={
                            "action_requests": [
                                {
                                    "name": "write_file",
                                    "args": tool_call["args"],
                                    "description": "Write notes.txt",
                                }
                            ]
                        },
                    ),
                ),
            }

        async def resume_request(
            self,
            decisions: dict[str, Any],
            conversation_id: int,
            on_text_chunk: Callable[[str, str], None] | None = None,
        ) -> dict[str, Any]:
            captured["decisions"] = decisions
            captured["conversation_id"] = conversation_id
            assert on_text_chunk is not None
            on_text_chunk("File ", "model:2")
            on_text_chunk("written.", "model:2")
            return {"messages": [AIMessage("File written.")]}

    class FakeConversationStore:
        async def mark_used(self, conversation_id: int, question: str) -> None:
            captured["marked_used"] = conversation_id, question

    cli = ChatCLI(
        cast(WorkspaceAgent, FakeAgent()),
        cast(ConversationStore, FakeConversationStore()),
        Terminal(use_color=False),
    )
    cli._conversation_id = 7

    def approve(prompt: str) -> str:
        captured["approval_prompt"] = prompt
        return "y"

    monkeypatch.setattr("builtins.input", approve)

    asyncio.run(cli._send_message("write a note"))

    assert captured["decisions"] == {
        "interrupt-1": {"decisions": [{"type": "approve"}]}
    }
    assert captured["conversation_id"] == 7
    assert captured["marked_used"] == (7, "write a note")
    assert captured["approval_prompt"] == (
        "Agent wants to write to file 'notes.txt'. Approve? [y/N] › "
    )
    assert capsys.readouterr().out == "\nAGENT\n  File written.\n"

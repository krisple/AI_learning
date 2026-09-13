import asyncio
from typing import cast

from langchain.messages import AIMessage, HumanMessage, ToolMessage

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

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain.tools import tool
from langchain_core.tools import BaseTool

from workspace_agent import filesystem_tools


@tool
def read_file() -> str:
    """Read a file for the test."""
    return "contents"


def test_get_filesystem_tools_uses_the_requested_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events: list[str] = []

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            events.append("session opened")
            return self

        async def __aexit__(self, *args: object) -> None:
            events.append("session closed")

    class FakeMCPClient:
        def __init__(self, connections: dict[str, Any]) -> None:
            captured["connections"] = connections

        def session(self, server_name: str) -> FakeSessionContext:
            captured["server_name"] = server_name
            return FakeSessionContext()

    async def fake_load_mcp_tools(session: object) -> list[BaseTool]:
        events.append("tools loaded")
        return [read_file]

    monkeypatch.setattr(filesystem_tools, "MultiServerMCPClient", FakeMCPClient)
    monkeypatch.setattr(filesystem_tools, "load_mcp_tools", fake_load_mcp_tools)

    async def use_tools() -> list[BaseTool]:
        async with filesystem_tools.get_filesystem_tools("/tmp/workspace") as tools:
            assert events == ["session opened", "tools loaded"]
            events.append("request handled")
            return tools

    result = asyncio.run(use_tools())

    assert result == [read_file]
    assert events == [
        "session opened",
        "tools loaded",
        "request handled",
        "session closed",
    ]
    assert captured["server_name"] == "filesystem"

    connection = captured["connections"]["filesystem"]
    assert connection["args"][-1] == "/tmp/workspace"

    list_roots = connection["session_kwargs"]["list_roots_callback"]
    roots = asyncio.run(list_roots(object()))
    assert str(roots.roots[0].uri) == Path("/tmp/workspace").resolve().as_uri()

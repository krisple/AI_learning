from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools


@asynccontextmanager
async def get_filesystem_tools(workspace_path: str) -> AsyncIterator[list[BaseTool]]:
    """Keep one filesystem MCP session open for the surrounding request."""
    connections: dict[str, Connection] = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                workspace_path,
            ],
        }
    }

    mcp_client = MultiServerMCPClient(connections)

    async with mcp_client.session("filesystem") as filesystem_session:
        yield await load_mcp_tools(filesystem_session)

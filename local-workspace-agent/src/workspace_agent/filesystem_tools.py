from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.types import ListRootsResult, Root


async def _list_workspace_roots(
    _context: object,
    *,
    workspace_root: Root,
) -> ListRootsResult:
    return ListRootsResult(roots=[workspace_root])


@asynccontextmanager
async def get_filesystem_tools(workspace_path: str) -> AsyncIterator[list[BaseTool]]:
    """Keep one filesystem MCP session open for the surrounding request."""
    workspace_root = Root.model_validate(
        {
            "uri": Path(workspace_path).resolve().as_uri(),
            "name": "workspace",
        }
    )

    connections: dict[str, Connection] = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                workspace_path,
            ],
            "session_kwargs": {
                "list_roots_callback": partial(
                    _list_workspace_roots,
                    workspace_root=workspace_root,
                )
            },
        }
    }

    mcp_client = MultiServerMCPClient(connections)

    async with mcp_client.session("filesystem") as filesystem_session:
        yield await load_mcp_tools(filesystem_session)

# Workspace Agent

A small learning project that uses LangChain, Ollama, SQLite checkpoints, and an
MCP filesystem server to interact with a local workspace.

## Run

Requires Python 3.13, `uv`, Node.js, and Ollama with the `qwen3.5:9b` model.

```bash
uv sync
uv run workspace-agent
```

Chats are saved locally. Available commands:

```text
/new
/chats [limit]
/resume <id> [messages]
/delete <id>
/history [messages]
/help
/exit
```

The agent can modify the workspace, but every write operation requires approval.

## Checks

```bash
uv run ruff check .
uv run pytest
```

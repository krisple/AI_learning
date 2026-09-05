# Workspace Agent

A small learning project that uses LangChain, Ollama, and an MCP filesystem
server to interact with a local workspace.

## Run

Requires Python 3.13, `uv`, Node.js, and Ollama with the `qwen3.5:9b` model.

```bash
uv sync
uv run workspace-agent
```

Type `exit` or `quit` to stop.

## Checks

```bash
uv run ruff check .
uv run pytest
```

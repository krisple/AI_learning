import asyncio
from pathlib import Path

from workspace_agent.agent import WorkspaceAgent
from workspace_agent.mode import AgentMode

DEFAULT_MODEL = "qwen3.5:9b"


async def run_chat() -> None:
    """Run the interactive terminal chat."""

    project_root = Path(__file__).resolve().parents[2]
    workspace_path = project_root / "workspace_agent_playground"

    agent = WorkspaceAgent(
        model_name=DEFAULT_MODEL,
        workspace_path=str(workspace_path),
        agent_mode=AgentMode.READ,
    )

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            return

        if question.lower() in {"exit", "quit"}: return

        if not question: continue

        response = await agent.run_request(question)

        print("Agent:", response["messages"][-1].content)


def main() -> None:
    """Start the command-line application."""
    
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nGoodbye!")

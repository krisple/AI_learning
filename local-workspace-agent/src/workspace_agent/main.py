import asyncio
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from workspace_agent.agent import WorkspaceAgent
from workspace_agent.cli import ChatCLI, Terminal
from workspace_agent.conversations import ConversationStore
from workspace_agent.mode import AgentMode

DEFAULT_MODEL = "qwen3.5:9b"


async def run_chat() -> None:
    """Create the application dependencies and run the terminal chat."""
    project_root = Path(__file__).resolve().parents[2]
    workspace_path = project_root / "workspace_agent_playground"
    database_path = project_root / ".data" / "conversations.sqlite"
    database_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        conversations = ConversationStore(checkpointer)
        await conversations.setup()

        agent = WorkspaceAgent(
            model_name=DEFAULT_MODEL,
            workspace_path=str(workspace_path),
            agent_mode=AgentMode.WRITE,
            checkpointer=checkpointer,
        )
        await ChatCLI(agent, conversations, Terminal()).run()


def main() -> None:
    """Start the command-line application."""
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nGoodbye!")

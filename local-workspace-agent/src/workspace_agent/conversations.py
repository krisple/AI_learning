from dataclasses import dataclass

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DEFAULT_TITLE = "New conversation"


@dataclass(frozen=True, slots=True)
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str


class ConversationStore:
    """Store conversation metadata alongside LangGraph checkpoints."""

    def __init__(self, checkpointer: AsyncSqliteSaver) -> None:
        self._checkpointer = checkpointer
        self._connection = checkpointer.conn

    async def setup(self) -> None:
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_agent_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._connection.commit()

    async def create(self) -> Conversation:
        cursor = await self._connection.execute(
            """
            INSERT INTO workspace_agent_conversations (title)
            VALUES (?)
            """,
            (DEFAULT_TITLE,),
        )
        await self._connection.commit()

        conversation_id = cursor.lastrowid
        await cursor.close()
        if conversation_id is None:
            raise RuntimeError("SQLite did not return a conversation ID")

        conversation = await self.get(conversation_id)
        if conversation is None:
            raise RuntimeError("The new conversation could not be loaded")
        return conversation

    async def get(self, conversation_id: int) -> Conversation | None:
        async with self._connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM workspace_agent_conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()

        return Conversation(*row) if row is not None else None

    async def list_recent(self, limit: int) -> list[Conversation]:
        async with self._connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM workspace_agent_conversations
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [Conversation(*row) for row in rows]

    async def mark_used(self, conversation_id: int, first_message: str) -> None:
        title = " ".join(first_message.split())[:60] or DEFAULT_TITLE
        await self._connection.execute(
            """
            UPDATE workspace_agent_conversations
            SET
                title = CASE WHEN title = ? THEN ? ELSE title END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (DEFAULT_TITLE, title, conversation_id),
        )
        await self._connection.commit()

    async def delete(self, conversation_id: int) -> bool:
        """Delete a conversation and all checkpoints associated with it."""
        if await self.get(conversation_id) is None:
            return False

        await self._checkpointer.adelete_thread(str(conversation_id))
        await self._connection.execute(
            "DELETE FROM workspace_agent_conversations WHERE id = ?",
            (conversation_id,),
        )
        await self._connection.commit()
        return True

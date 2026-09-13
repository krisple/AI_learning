import asyncio
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from workspace_agent.conversations import DEFAULT_TITLE, ConversationStore


def test_conversation_store_creates_lists_and_updates_threads(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "conversations.sqlite"

        async with AsyncSqliteSaver.from_conn_string(
            str(database_path)
        ) as checkpointer:
            store = ConversationStore(checkpointer)
            await store.setup()

            first = await store.create()
            second = await store.create()

            assert first.title == DEFAULT_TITLE
            assert second.id == first.id + 1

            await store.mark_used(first.id, "  Explain   Python decorators  ")
            updated = await store.get(first.id)

            assert updated is not None
            assert updated.title == "Explain Python decorators"
            recent_ids = {item.id for item in await store.list_recent(2)}
            assert recent_ids == {first.id, second.id}
            assert len(await store.list_recent(1)) == 1

            config: RunnableConfig = {
                "configurable": {
                    "thread_id": str(first.id),
                    "checkpoint_ns": "",
                }
            }
            await checkpointer.aput(config, empty_checkpoint(), {}, {})
            assert [checkpoint async for checkpoint in checkpointer.alist(config)]

            assert await store.delete(first.id) is True
            assert await store.get(first.id) is None
            assert [checkpoint async for checkpoint in checkpointer.alist(config)] == []
            assert await store.delete(first.id) is False

    asyncio.run(scenario())

import sys

from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage

from workspace_agent.agent import WorkspaceAgent
from workspace_agent.conversations import ConversationStore

DEFAULT_CHAT_LIMIT = 10
DEFAULT_HISTORY_LIMIT = 10
MAX_DISPLAY_LIMIT = 100


class CommandError(ValueError):
    """Raised when a slash command has invalid arguments."""


class Terminal:
    """Render readable, optionally colored terminal output."""

    RESET = "\033[0m"
    BLUE = "\033[1;36m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    DIM = "\033[2m"

    def __init__(self, use_color: bool | None = None) -> None:
        self._use_color = sys.stdout.isatty() if use_color is None else use_color

    def prompt(self, conversation_id: int | None) -> str:
        chat = f"chat #{conversation_id}" if conversation_id else "no active chat"
        return input(self._paint(f"YOU [{chat}] › ", self.BLUE))

    def show_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            label, color = "YOU", self.BLUE
        elif isinstance(message, AIMessage):
            label, color = "AGENT", self.GREEN
        else:
            return

        text = str(message.text).strip()
        if not text:
            return

        print(f"\n{self._paint(label, color)}")
        for line in text.splitlines():
            print(f"  {line}")

    def info(self, message: str) -> None:
        print(self._paint(message, self.YELLOW))

    def error(self, message: str) -> None:
        print(self._paint(f"Error: {message}", self.RED))

    def muted(self, message: str) -> None:
        print(self._paint(message, self.DIM))

    def _paint(self, text: str, color: str) -> str:
        if not self._use_color:
            return text
        return f"{color}{text}{self.RESET}"


class ChatCLI:
    """Interactive slash-command interface for persisted conversations."""

    def __init__(
        self,
        agent: WorkspaceAgent,
        conversations: ConversationStore,
        terminal: Terminal,
    ) -> None:
        self._agent = agent
        self._conversations = conversations
        self._terminal = terminal
        self._conversation_id: int | None = None

    async def run(self) -> None:
        self._terminal.info("Workspace Agent")
        self._terminal.muted("Type /help for commands. Your chats are saved locally.")

        while True:
            try:
                user_input = self._terminal.prompt(self._conversation_id).strip()
            except EOFError:
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not await self._handle_command(user_input):
                    return
                continue

            await self._send_message(user_input)

    async def _handle_command(self, raw_command: str) -> bool:
        command, *arguments = raw_command.split()
        command = command.lower()

        try:
            if command in {"/exit", "/quit"}:
                self._expect_no_arguments(command, arguments)
                return False
            if command == "/help":
                self._expect_no_arguments(command, arguments)
                self._show_help()
            elif command == "/new":
                self._expect_no_arguments(command, arguments)
                await self._new_conversation()
            elif command == "/chats":
                limit = self._optional_limit(arguments, DEFAULT_CHAT_LIMIT, "/chats")
                await self._show_chats(limit)
            elif command == "/history":
                limit = self._optional_limit(
                    arguments,
                    DEFAULT_HISTORY_LIMIT,
                    "/history",
                )
                await self._show_history(limit)
            elif command == "/resume":
                await self._resume(arguments)
            elif command == "/delete":
                await self._delete(arguments)
            else:
                raise CommandError(f"unknown command: {command}. Type /help.")
        except CommandError as error:
            self._terminal.error(str(error))

        return True

    async def _new_conversation(self) -> None:
        conversation = await self._conversations.create()
        self._conversation_id = conversation.id
        self._terminal.info(f"Started conversation #{conversation.id}.")

    async def _show_chats(self, limit: int) -> None:
        conversations = await self._conversations.list_recent(limit)
        if not conversations:
            self._terminal.muted("No saved conversations yet.")
            return

        self._terminal.info("Recent conversations")
        for conversation in conversations:
            marker = "*" if conversation.id == self._conversation_id else " "
            updated_at = conversation.updated_at[:16]
            print(f"{marker} #{conversation.id:<4} {updated_at}  {conversation.title}")

    async def _show_history(self, limit: int) -> None:
        if self._conversation_id is None:
            raise CommandError("no active conversation. Use /new or /resume <id>.")

        messages = await self._agent.get_conversation_messages(self._conversation_id)
        recent_messages = _recent_visible_messages(messages, limit)
        if not recent_messages:
            self._terminal.muted("This conversation has no messages yet.")
            return

        self._terminal.info(
            f"Last {len(recent_messages)} messages (oldest to newest) "
            f"from conversation #{self._conversation_id}"
        )
        for message in recent_messages:
            self._terminal.show_message(message)

    async def _resume(self, arguments: list[str]) -> None:
        if not 1 <= len(arguments) <= 2:
            raise CommandError("usage: /resume <id> [messages]")

        conversation_id = self._positive_integer(arguments[0], "conversation ID")
        history_limit = (
            self._positive_integer(arguments[1], "message limit")
            if len(arguments) == 2
            else DEFAULT_HISTORY_LIMIT
        )
        self._validate_limit(history_limit)

        conversation = await self._conversations.get(conversation_id)
        if conversation is None:
            raise CommandError(f"conversation #{conversation_id} does not exist.")

        self._conversation_id = conversation.id
        self._terminal.info(
            f"Resumed conversation #{conversation.id}: {conversation.title}"
        )
        await self._show_history(history_limit)

    async def _delete(self, arguments: list[str]) -> None:
        if len(arguments) != 1:
            raise CommandError("usage: /delete <id>")

        conversation_id = self._positive_integer(arguments[0], "conversation ID")
        if not await self._conversations.delete(conversation_id):
            raise CommandError(f"conversation #{conversation_id} does not exist.")

        if self._conversation_id == conversation_id:
            self._conversation_id = None

        self._terminal.info(f"Deleted conversation #{conversation_id}.")

    async def _send_message(self, question: str) -> None:
        if self._conversation_id is None:
            await self._new_conversation()

        assert self._conversation_id is not None
        response = await self._agent.run_request(question, self._conversation_id)
        await self._conversations.mark_used(self._conversation_id, question)

        last_message = response["messages"][-1]
        if isinstance(last_message, BaseMessage):
            self._terminal.show_message(last_message)
        else:
            self._terminal.error("the agent returned an unsupported message format.")

    def _show_help(self) -> None:
        print(
            """
/new                    Start a new conversation
/chats [limit]          List recent conversations
/resume <id> [messages] Resume a conversation and show recent messages
/delete <id>            Delete a conversation and its history
/history [messages]     Show messages from the active conversation
/help                   Show this help
/exit                   Exit
""".strip()
        )

    @classmethod
    def _optional_limit(
        cls,
        arguments: list[str],
        default: int,
        command: str,
    ) -> int:
        if len(arguments) > 1:
            raise CommandError(f"usage: {command} [limit]")
        if not arguments:
            return default

        limit = cls._positive_integer(arguments[0], "limit")
        cls._validate_limit(limit)
        return limit

    @staticmethod
    def _positive_integer(value: str, name: str) -> int:
        try:
            parsed_value = int(value)
        except ValueError as error:
            raise CommandError(f"{name} must be a positive integer.") from error

        if parsed_value < 1:
            raise CommandError(f"{name} must be a positive integer.")
        return parsed_value

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit > MAX_DISPLAY_LIMIT:
            raise CommandError(f"limit cannot be greater than {MAX_DISPLAY_LIMIT}.")

    @staticmethod
    def _expect_no_arguments(command: str, arguments: list[str]) -> None:
        if arguments:
            raise CommandError(f"usage: {command}")


def _visible_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [
        message
        for message in messages
        if (
            isinstance(message, HumanMessage)
            or isinstance(message, AIMessage)
            and not message.tool_calls
        )
        and str(message.text).strip()
    ]


def _recent_visible_messages(
    messages: list[BaseMessage], limit: int
) -> list[BaseMessage]:
    """Select recent messages without reversing their chronological order."""
    return _visible_messages(messages)[-limit:]

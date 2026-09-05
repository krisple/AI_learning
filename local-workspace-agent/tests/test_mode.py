from workspace_agent.mode import AgentMode


def test_agent_modes_have_readable_values() -> None:
    assert AgentMode.READ.value == "read"
    assert AgentMode.WRITE.value == "write"

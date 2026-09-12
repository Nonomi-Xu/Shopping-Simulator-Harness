"""Public single-run entry point backed by the seven-part AgentOrchestrator."""

from .contracts import RUNTIME_VERSION, AgentRunResult, RunStatus, RunTerminalReason
from .orchestrator import AgentOrchestrator, HarnessRuntime

AGENT_RUNNER_VERSION = RUNTIME_VERSION


class AgentRunner(AgentOrchestrator):
    """Stable Python/CLI entry point; execution is owned by AgentOrchestrator."""

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class RunState:
    """Control state for one Agent run; messages remain the full trajectory."""

    step: int = 0  # Completed LLM calls, including the final-answer call.
    tool_rounds: int = 0
    stop_reason: str = ''
    continuation: object | None = None
    started_at: float = field(default_factory=monotonic, repr=False)
    finished_at: float | None = field(default=None, repr=False)

    @property
    def elapsed(self):
        end = self.finished_at if self.finished_at is not None else monotonic()
        return end - self.started_at

    def finish(self, reason):
        self.stop_reason = reason
        self.finished_at = monotonic()


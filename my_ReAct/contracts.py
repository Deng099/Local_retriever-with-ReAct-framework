from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolError:
    type: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data=None, metadata=None):
        return cls(ok=True, data=data, metadata=metadata or {})

    @classmethod
    def failure(
        cls,
        error_type,
        message,
        *,
        retryable=False,
        details=None,
        metadata=None,
    ):
        return cls(
            ok=False,
            error=ToolError(
                type=error_type,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
            metadata=metadata or {},
        )

    def to_dict(self):
        return {
            'ok': self.ok,
            'data': self.data,
            'error': self.error.to_dict() if self.error else None,
            'metadata': self.metadata,
        }


@dataclass(frozen=True)
class AgentRunResult:
    status: str
    output: str | None
    termination_reason: str
    error: ToolError | None = None

    @classmethod
    def completed(cls, output, termination_reason='final_answer'):
        return cls(
            status='completed',
            output=output,
            termination_reason=termination_reason,
        )

    @classmethod
    def incomplete(cls, termination_reason, output=None):
        return cls(
            status='incomplete',
            output=output,
            termination_reason=termination_reason,
        )

    @classmethod
    def failed(cls, error_type, message, termination_reason='runtime_error'):
        return cls(
            status='error',
            output=None,
            termination_reason=termination_reason,
            error=ToolError(type=error_type, message=message),
        )

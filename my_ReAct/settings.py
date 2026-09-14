import os
from dataclasses import dataclass, field


def env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be a boolean value')


@dataclass(frozen=True)
class LLMSettings:
    model: str
    api_key: str = field(repr=False)
    base_url: str = 'https://api.openai.com/v1'
    api_type: str = 'chat'
    reasoning_effort: str | None = None
    trust_env: bool = True

    @classmethod
    def from_env(cls):
        model = os.getenv('OPENAI_MODEL')
        api_key = os.getenv('OPENAI_API_KEY')
        if not model:
            raise ValueError('OPENAI_MODEL is not set')
        if not api_key:
            raise ValueError('OPENAI_API_KEY is not set')
        api_type = os.getenv('OPENAI_API_TYPE', 'chat').lower()
        if api_type not in {'chat', 'responses'}:
            raise ValueError("OPENAI_API_TYPE must be 'chat' or 'responses'")
        return cls(
            model=model,
            api_key=api_key,
            base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            api_type=api_type,
            reasoning_effort=os.getenv('OPENAI_REASONING_EFFORT'),
            trust_env=env_flag('OPENAI_TRUST_ENV'),
        )


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int = 10
    python_execution_timeout: float = 30.0
    python_output_limit: int = 20_000

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError('max_steps must be at least 1')
        if self.python_execution_timeout <= 0:
            raise ValueError('python_execution_timeout must be positive')
        if self.python_output_limit < 1:
            raise ValueError('python_output_limit must be positive')

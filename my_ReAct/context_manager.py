from abc import ABC, abstractmethod
import json

from my_ReAct.run_state import RunState


class ContextManager(ABC):
    """Transform a trajectory copy before each LLM call, including finalization."""

    @abstractmethod
    def process(self, messages: list[dict], state: RunState) -> list[dict]:
        raise NotImplementedError

    def reset(self):
        """Reset any per-run counters or memory before the first call."""


class RecentTurnsContextManager(ContextManager):
    """Keep the initial instructions/task and recent assistant/tool rounds.

    This is a turn window, not a token budget. A single long observation can
    still exceed a model's context limit. Tool calls and outputs stay together.
    """

    def __init__(self, keep_recent_turns=4):
        if isinstance(keep_recent_turns, bool) or not isinstance(keep_recent_turns, int) or keep_recent_turns < 1:
            raise ValueError('keep_recent_turns must be a positive integer')
        self.keep_recent_turns = keep_recent_turns

    def process(self, messages, state):
        starts = [i for i, message in enumerate(messages) if message['role'] == 'assistant']
        if len(starts) <= self.keep_recent_turns:
            return messages
        # Each block starts with an assistant and includes every associated
        # tool response. The trailing budget-exhausted user message also stays.
        retained_start = starts[-self.keep_recent_turns]
        return messages[:starts[0]] + [{
            'role': 'user',
            'content': '[Earlier interaction rounds omitted from this context; their evidence is not shown.]',
        }] + messages[retained_start:]


class BudgetContextManager(ContextManager):
    """Bound the visible trajectory while retaining recent evidence.

    ``ReActAgent.messages`` remains the complete audit trail. This manager only
    builds a copy for the next model request, so compaction cannot destroy the
    data saved in ``runs/``. The budget is deliberately measured in characters
    to keep this dependency-free and portable; callers should leave headroom
    for provider-specific tokenization and the model's output.
    """

    def __init__(
        self,
        max_chars=120_000,
        keep_recent_turns=4,
        max_search_snippet_chars=320,
        max_visit_content_chars=2_000,
        max_extraction_chars=6_000,
        max_python_chars=4_000,
        max_tool_chars=16_000,
    ):
        for name, value in (
            ('max_chars', max_chars),
            ('keep_recent_turns', keep_recent_turns),
            ('max_search_snippet_chars', max_search_snippet_chars),
            ('max_visit_content_chars', max_visit_content_chars),
            ('max_extraction_chars', max_extraction_chars),
            ('max_python_chars', max_python_chars),
            ('max_tool_chars', max_tool_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.max_chars = max_chars
        self.keep_recent_turns = keep_recent_turns
        self.max_search_snippet_chars = max_search_snippet_chars
        self.max_visit_content_chars = max_visit_content_chars
        self.max_extraction_chars = max_extraction_chars
        self.max_python_chars = max_python_chars
        self.max_tool_chars = max_tool_chars
        self.stats = {
            'calls': 0,
            'last_input_chars': 0,
            'last_view_chars': 0,
            'last_omitted_rounds': 0,
            'last_truncated': False,
            'max_view_chars': 0,
            'total_omitted_rounds': 0,
        }

    def reset(self):
        self.stats = {
            'calls': 0,
            'last_input_chars': 0,
            'last_view_chars': 0,
            'last_omitted_rounds': 0,
            'last_truncated': False,
            'max_view_chars': 0,
            'total_omitted_rounds': 0,
        }

    def _record_stats(self, input_chars, view, omitted_rounds=0):
        view_chars = self._size(view)
        self.stats.update({
            'calls': self.stats['calls'] + 1,
            'last_input_chars': input_chars,
            'last_view_chars': view_chars,
            'last_omitted_rounds': omitted_rounds,
            'last_truncated': view_chars < input_chars,
            'max_view_chars': max(self.stats['max_view_chars'], view_chars),
            'total_omitted_rounds': self.stats['total_omitted_rounds'] + omitted_rounds,
        })

    @staticmethod
    def _size(messages):
        return len(json.dumps(messages, ensure_ascii=False, separators=(',', ':')))

    @staticmethod
    def _clip(value, limit):
        if not isinstance(value, str) or len(value) <= limit:
            return value
        return value[:limit] + f'\n[… context view clipped; original {len(value)} chars …]'

    def _compact_value(self, value, string_limit=800, depth=0):
        if isinstance(value, str):
            return self._clip(value, string_limit)
        if isinstance(value, list):
            # Tool payloads are normally short. This guard handles an
            # accidentally huge custom tool without introducing a schema.
            items = [self._compact_value(item, string_limit, depth + 1) for item in value[:20]]
            if len(value) > 20:
                items.append(f'[… {len(value) - 20} list items omitted …]')
            return items
        if isinstance(value, dict):
            return {
                key: self._compact_value(item, string_limit, depth + 1)
                for key, item in value.items()
            }
        return value

    def _compact_tool_content(self, message):
        name = message.get('name')
        initial_limit = 800
        if name == 'visit':
            initial_limit = max(self.max_visit_content_chars, self.max_extraction_chars)
        content = self._compact_value(message.get('content'), string_limit=initial_limit)
        if isinstance(content, dict):
            data = content.get('data')
            if name == 'search' and isinstance(data, dict):
                for result in data.get('results', []):
                    for chunk in result.get('chunks', []) if isinstance(result, dict) else []:
                        if isinstance(chunk, dict):
                            if 'snippet' in chunk:
                                chunk['snippet'] = self._clip(
                                    chunk['snippet'], self.max_search_snippet_chars
                                )
                            if 'text' in chunk:
                                chunk['text'] = self._clip(
                                    chunk['text'], self.max_search_snippet_chars
                                )
            elif name == 'visit' and isinstance(data, dict):
                document = data.get('document')
                if isinstance(document, dict) and 'content' in document:
                    document['content'] = self._clip(
                        document['content'], self.max_visit_content_chars
                    )
                extraction = data.get('extraction')
                if isinstance(extraction, dict) and 'content' in extraction:
                    extraction['content'] = self._clip(
                        extraction['content'], self.max_extraction_chars
                    )
            elif name == 'python' and isinstance(data, dict) and 'result' in data:
                data['result'] = self._clip(data['result'], self.max_python_chars)

        if self._size([content]) <= self.max_tool_chars:
            return content
        compacted = self._compact_value(content, string_limit=240)
        if self._size([compacted]) <= self.max_tool_chars:
            return compacted
        # Preserve the envelope and an actionable indication that the view is
        # incomplete. The complete result remains in the trajectory on disk.
        if isinstance(content, dict):
            return {
                'ok': content.get('ok'),
                'error': content.get('error'),
                'context_truncated': True,
                'summary': 'Tool observation exceeded the context view budget; use its source/reference again.',
            }
        return {
            'context_truncated': True,
            'summary': 'Tool observation exceeded the context view budget; use the tool again if needed.',
        }

    def _compact_messages(self, messages):
        compacted = []
        for message in messages:
            item = dict(message)
            if item.get('role') == 'tool':
                item['content'] = self._compact_tool_content(item)
            compacted.append(item)
        return compacted

    def process(self, messages, state: RunState):
        input_chars = self._size(messages)
        messages = self._compact_messages(messages)
        if self._size(messages) <= self.max_chars:
            self._record_stats(input_chars, messages)
            return messages

        assistant_indices = [
            index for index, message in enumerate(messages)
            if message.get('role') == 'assistant'
        ]
        if not assistant_indices:
            self._record_stats(input_chars, messages)
            return messages

        prefix = messages[:assistant_indices[0]]
        blocks = []
        for position, start in enumerate(assistant_indices):
            end = assistant_indices[position + 1] if position + 1 < len(assistant_indices) else len(messages)
            blocks.append(messages[start:end])

        def flatten(block_list):
            return [message for block in block_list for message in block]

        # Keep the tail user instruction (for example, the exhausted-budget
        # instruction) with the latest block; older blocks are removed whole so
        # assistant/tool call pairs remain valid for every provider.
        retained = list(blocks)
        # Prefer the configured recent window when a choice is necessary, then
        # continue dropping older complete blocks until the hard budget fits.
        while len(retained) > self.keep_recent_turns and self._size(prefix + flatten(retained)) > self.max_chars:
            retained.pop(0)
        while retained and self._size(prefix + flatten(retained)) > self.max_chars:
            retained.pop(0)

        omitted = len(blocks) - len(retained)
        if omitted:
            prefix = prefix + [{
                'role': 'user',
                'content': f'[Earlier rounds omitted from this context view: {omitted}. '
                           'Their full observations remain in the trajectory and can be re-read by reference.]',
            }]

        view = prefix + flatten(retained)
        # A very large system prompt is unusual, but make the invariant clear
        # instead of silently violating max_chars.
        if self._size(view) > self.max_chars and retained:
            while retained and self._size(prefix + flatten(retained)) > self.max_chars:
                retained.pop(0)
            view = prefix + flatten(retained)
            omitted = len(blocks) - len(retained)
        self._record_stats(input_chars, view, omitted)
        return view

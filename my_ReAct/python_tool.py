from my_ReAct.base_tool import BaseTool
from my_ReAct.contracts import ToolResult
from my_ReAct.python_session import PythonSessionManager


class PythonTool(BaseTool):
    '''Task-scoped client for one managed Python worker; it is not a security sandbox.'''

    def __init__(
        self,
        session_manager=None,
        execution_timeout=30.0,
        output_limit=20_000,
    ):
        super().__init__(
            name='python',
            description='Execute Python for calculations and data processing. Variables persist across calls in this task.',
            parameters={
                'type': 'object',
                'properties': {'code': {'type': 'string'}},
                'required': ['code'],
            },
        )
        self._owns_manager = session_manager is None
        self.session_manager = session_manager or PythonSessionManager(
            execution_timeout=execution_timeout,
            stdout_limit=output_limit,
            stderr_limit=output_limit,
            result_limit=output_limit,
        )
        self.session_id = self.session_manager.create_session()
        self._closed = False

    def _execute(self, code):
        if self._closed:
            raise RuntimeError('Python tool session is closed')
        result = self.session_manager.execute(self.session_id, code)
        if result['status'] == 'error':
            return ToolResult.failure(
                result['error']['type'],
                result['error']['message'],
                details={
                    'stdout': result['stdout'],
                    'stderr': result['stderr'],
                    'traceback': result['error']['traceback'],
                },
                metadata={'truncated': result['truncated']},
            )
        return ToolResult.success({
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'result': result['result'],
        }, metadata={'truncated': result['truncated']})

    def close(self):
        if self._closed:
            return
        self.session_manager.close_session(self.session_id)
        if self._owns_manager:
            self.session_manager.close_all()
        self._closed = True

    @property
    def is_active(self):
        return not self._closed and self.session_manager.is_active(self.session_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()

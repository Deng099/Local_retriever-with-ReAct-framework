import ast
import multiprocessing
import os
import reprlib
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from threading import Lock


class PythonSessionError(RuntimeError):
    pass


class PythonSessionNotFound(PythonSessionError):
    pass


class PythonSessionTimeout(PythonSessionError):
    pass


class PythonSessionCrashed(PythonSessionError):
    pass


class _LimitedStringIO(StringIO):
    def __init__(self, max_chars):
        super().__init__()
        self.max_chars = max_chars
        self.total_chars = 0

    def write(self, text):
        self.total_chars += len(text)
        remaining = max(0, self.max_chars - self.tell())
        if remaining:
            super().write(text[:remaining])
        return len(text)

    @property
    def truncated(self):
        return self.total_chars > self.max_chars


def _default_namespace():
    import collections
    import datetime
    import functools
    import itertools
    import json
    import math
    import re
    import statistics

    namespace = {
        'collections': collections,
        'datetime': datetime,
        'functools': functools,
        'itertools': itertools,
        'json': json,
        'math': math,
        're': re,
        'statistics': statistics,
    }
    try:
        import numpy as np
        namespace['np'] = np
    except ImportError:
        pass
    try:
        import pandas as pd
        namespace['pd'] = pd
    except ImportError:
        pass
    try:
        import scipy
        namespace['scipy'] = scipy
    except ImportError:
        pass
    return namespace


def _bounded_repr(value, max_chars):
    renderer = reprlib.Repr()
    renderer.maxstring = max_chars
    renderer.maxother = max_chars
    renderer.maxlist = 100
    renderer.maxtuple = 100
    renderer.maxdict = 100
    rendered = renderer.repr(value)
    summarized = (
        (isinstance(value, (str, bytes)) and len(value) + 2 > max_chars)
        or (isinstance(value, list) and len(value) > renderer.maxlist)
        or (isinstance(value, tuple) and len(value) > renderer.maxtuple)
        or (isinstance(value, dict) and len(value) > renderer.maxdict)
    )
    return rendered[:max_chars], summarized or len(rendered) > max_chars


def _execute_code(code, namespace, stdout_limit, stderr_limit, result_limit):
    stdout = _LimitedStringIO(stdout_limit)
    stderr = _LimitedStringIO(stderr_limit)
    try:
        tree = ast.parse(code, mode='exec')
        final_expression = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            final_expression = ast.Expression(tree.body.pop().value)

        with redirect_stdout(stdout), redirect_stderr(stderr):
            if tree.body:
                exec(compile(tree, '<python-tool>', 'exec'), namespace, namespace)
            result = None
            if final_expression is not None:
                result = eval(
                    compile(final_expression, '<python-tool>', 'eval'),
                    namespace,
                    namespace,
                )
        rendered_result, result_truncated = (
            (None, False)
            if result is None
            else _bounded_repr(result, result_limit)
        )
        return {
            'status': 'ok',
            'stdout': stdout.getvalue(),
            'stderr': stderr.getvalue(),
            'result': rendered_result,
            'error': None,
            'truncated': {
                'stdout': stdout.truncated,
                'stderr': stderr.truncated,
                'result': result_truncated,
            },
        }
    except Exception as exc:
        return {
            'status': 'error',
            'stdout': stdout.getvalue(),
            'stderr': stderr.getvalue(),
            'result': None,
            'error': {
                'type': type(exc).__name__,
                'message': str(exc),
                'traceback': traceback.format_exc(),
            },
            'truncated': {
                'stdout': stdout.truncated,
                'stderr': stderr.truncated,
                'result': False,
            },
        }


def _python_worker(connection, stdout_limit, stderr_limit, result_limit):
    namespace = _default_namespace()
    try:
        while True:
            message = connection.recv()
            operation = message.get('operation')
            if operation == 'execute':
                connection.send(_execute_code(
                    message['code'],
                    namespace,
                    stdout_limit,
                    stderr_limit,
                    result_limit,
                ))
            elif operation == 'close':
                return
            else:
                connection.send({
                    'status': 'error',
                    'stdout': '',
                    'stderr': '',
                    'result': None,
                    'error': {
                        'type': 'ValueError',
                        'message': f'Unknown worker operation: {operation}',
                        'traceback': '',
                    },
                    'truncated': {'stdout': False, 'stderr': False, 'result': False},
                })
    except EOFError:
        return
    finally:
        connection.close()


@dataclass
class _PythonSession:
    process: object
    connection: object
    last_used: float
    lock: Lock = field(default_factory=Lock)


class PythonSessionManager:
    '''Own persistent Python worker processes keyed by task-scoped session ids.'''

    def __init__(
        self,
        execution_timeout=30.0,
        close_timeout=1.0,
        stdout_limit=20_000,
        stderr_limit=20_000,
        result_limit=20_000,
    ):
        self.execution_timeout = execution_timeout
        self.close_timeout = close_timeout
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit
        self.result_limit = result_limit
        if min(stdout_limit, stderr_limit, result_limit) < 1:
            raise ValueError('Python output limits must be positive')
        self._context = multiprocessing.get_context('spawn')
        self._sessions = {}
        self._lock = Lock()

    def create_session(self, session_id=None):
        session_id = session_id or uuid.uuid4().hex
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f'Python session already exists: {session_id}')
            parent_connection, child_connection = self._context.Pipe()
            process = self._context.Process(
                target=_python_worker,
                args=(
                    child_connection,
                    self.stdout_limit,
                    self.stderr_limit,
                    self.result_limit,
                ),
                name=f'python-tool-{session_id[:12]}',
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._sessions[session_id] = _PythonSession(
                process=process,
                connection=parent_connection,
                last_used=time.monotonic(),
            )
        return session_id

    def execute(self, session_id, code, timeout=None):
        session = self._get_session(session_id)
        execution_timeout = self.execution_timeout if timeout is None else timeout
        with session.lock:
            if not session.process.is_alive():
                exit_code = session.process.exitcode
                self._drop_session(session_id, session)
                raise PythonSessionCrashed(
                    f'Python session {session_id} is not running (exit code {exit_code})'
                )
            try:
                session.connection.send({'operation': 'execute', 'code': code})
                if not session.connection.poll(execution_timeout):
                    self._drop_session(session_id, session, terminate=True)
                    raise PythonSessionTimeout(
                        f'Python execution exceeded {execution_timeout} seconds; session was closed'
                    )
                result = session.connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                exit_code = session.process.exitcode
                self._drop_session(session_id, session, terminate=True)
                raise PythonSessionCrashed(
                    f'Python session {session_id} crashed (exit code {exit_code})'
                ) from exc
            session.last_used = time.monotonic()
            return result

    def close_session(self, session_id):
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        with session.lock:
            self._close_worker(session)
        return True

    def cleanup_idle(self, max_idle_seconds):
        cutoff = time.monotonic() - max_idle_seconds
        with self._lock:
            session_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.last_used <= cutoff
            ]
        for session_id in session_ids:
            self.close_session(session_id)
        return session_ids

    def close_all(self):
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.close_session(session_id)

    def is_active(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
            return bool(session and session.process.is_alive())

    def worker_pid(self, session_id):
        return self._get_session(session_id).process.pid

    def _get_session(self, session_id):
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise PythonSessionNotFound(f'Unknown Python session: {session_id}') from exc

    def _drop_session(self, session_id, session, terminate=False):
        with self._lock:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id)
        if terminate and session.process.is_alive():
            session.process.terminate()
        session.process.join(self.close_timeout)
        session.connection.close()

    def _close_worker(self, session):
        if session.process.is_alive():
            try:
                session.connection.send({'operation': 'close'})
            except (BrokenPipeError, OSError):
                pass
            session.process.join(self.close_timeout)
        if session.process.is_alive():
            session.process.terminate()
            session.process.join(self.close_timeout)
        session.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close_all()

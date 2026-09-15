"""Small helpers for validated, non-overwriting data artifact creation."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_output_paths(payload, metadata, inputs=()):
    outputs = [Path(payload), Path(metadata)]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != 2 or set(resolved) & {Path(path).resolve() for path in inputs}:
        raise ValueError('Output paths overlap each other or an input file')
    for path in outputs:
        if os.path.lexists(path):
            raise FileExistsError(f'Output already exists: {path}')


@contextmanager
def artifact_pair(payload, metadata, inputs=()):
    """Publish metadata first, payload last as completion marker.

    Hard links prevent replacement if a competing writer creates a destination.
    Exceptions roll back only this invocation's files. SIGKILL can leave orphan
    metadata/temp files, but cannot publish a payload before its metadata.
    """
    outputs = [Path(payload), Path(metadata)]
    check_output_paths(*outputs, inputs=inputs)
    temporary, published = [], []
    try:
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
            os.close(descriptor)
            temporary.append(Path(name))
        yield tuple(temporary)
        for path in temporary:
            with path.open('rb') as source:
                os.fsync(source.fileno())
        for position in (1, 0):
            os.link(temporary[position], outputs[position])
            published.append(outputs[position])
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def validate_chunk(chunk):
    if not isinstance(chunk, dict):
        raise ValueError('Chunk must be an object')
    for field in ('doc_id', 'chunk_id'):
        value = chunk.get(field)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f'Chunk requires a nonempty {field}')
    if isinstance(chunk['chunk_id'], int) and chunk['chunk_id'] < 0:
        raise ValueError('chunk_id must be nonnegative')
    if not isinstance(chunk.get('text'), str) or not chunk['text'].strip():
        raise ValueError('Chunk requires nonempty string text')

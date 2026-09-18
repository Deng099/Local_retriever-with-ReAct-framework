import argparse
import json
from pathlib import Path


def load_optional_mapping(path, *, list_key=None):
    if path is None:
        return {}
    with path.open(encoding='utf-8') as file:
        value = json.load(file)
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, list) and list_key:
        return {str(item[list_key]): item for item in value}
    raise ValueError(f'{path}: expected an object or a list with {list_key!r}')


def find_run_paths(path, query_id=None):
    paths = sorted(path.glob('run_*.json')) if path.is_dir() else [path]
    if query_id is not None:
        paths = [candidate for candidate in paths if candidate.stem == f'run_{query_id}']
    if not paths:
        raise FileNotFoundError(f'No run JSON files found at {path}')
    return paths


def token_totals(usages):
    prompt = sum(
        usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0
        for usage in usages
    )
    completion = sum(
        usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0
        for usage in usages
    )
    return prompt, completion


def recall(retrieved, relevant):
    relevant = {str(item) for item in relevant}
    if not relevant:
        return None
    return len(retrieved & relevant), len(relevant)


def compact_message(message, index, final_index, max_chars):
    role = message.get('role', 'unknown')
    name = message.get('name')
    latency = message.get('latency_seconds')
    heading = f'[{index}] {role}' + (f'/{name}' if name else '')
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        heading += f' ({latency:.3f}s)'
    if message.get('tool_calls'):
        content = json.dumps(message['tool_calls'], ensure_ascii=False)
    else:
        content = message.get('content')
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
    if index != final_index and len(content) > max_chars:
        content = content[:max_chars] + '...'
    return f'{heading}\n{content}'


def render_run(run, answers, relevance, show_trajectory=False, max_chars=500):
    query_id = str(run.get('query_id'))
    retrieved = {str(item) for item in run.get('retrieved_docids', [])}
    labels = relevance.get(query_id, {})
    evidence = recall(retrieved, labels.get('evidence', []))
    gold = recall(retrieved, labels.get('gold', []))
    prompt_tokens, completion_tokens = token_totals(run.get('usage', []))
    state = run.get('run_state') or {}
    timing = run.get('timing') or {}
    lines = [
        '=' * 80,
        f'ID: {query_id}',
        f"status: {run.get('status')}",
        f"reason: {run.get('termination_reason')}",
        f"elapsed: {state.get('elapsed')}s",
        f"steps/tool rounds: {state.get('step')}/{state.get('tool_rounds')}",
        f"tools: {run.get('tool_call_counts', {})}",
        f'retrieved unique docs: {len(retrieved)}',
        f'tokens (prompt/completion): {prompt_tokens}/{completion_tokens}',
        f"timing (llm/tools): {timing.get('llm_seconds')}/{timing.get('tool_seconds')}",
        'evidence recall: unavailable' if evidence is None else f'evidence recall: {evidence[0]}/{evidence[1]} ({100 * evidence[0] / evidence[1]:.1f}%)',
        'gold recall: unavailable' if gold is None else f'gold recall: {gold[0]}/{gold[1]} ({100 * gold[0] / gold[1]:.1f}%)',
        '',
        'MODEL OUTPUT:',
        (run.get('result') or [{}])[-1].get('output', ''),
    ]
    answer = answers.get(query_id)
    if isinstance(answer, dict):
        answer = answer.get('answer')
    if answer is not None:
        lines.extend(['', 'GROUND TRUTH:', str(answer)])
    if show_trajectory:
        trajectory = run.get('trajectory', [])
        lines.extend(['', 'TRAJECTORY:'])
        lines.extend(
            compact_message(message, index, len(trajectory) - 1, max_chars)
            for index, message in enumerate(trajectory)
        )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Inspect ReAct run files and optional benchmark labels')
    parser.add_argument('run_path', type=Path, help='A run JSON file or a directory containing run_*.json')
    parser.add_argument('--answers', type=Path, help='JSON list/object containing query_id and answer')
    parser.add_argument('--relevance', type=Path, help='JSON mapping query IDs to evidence/gold doc IDs')
    parser.add_argument('--query-id', help='Inspect one query from a run directory')
    parser.add_argument('--trajectory', action='store_true', help='Show a compact trajectory')
    parser.add_argument('--max-content-chars', type=int, default=500)
    args = parser.parse_args()
    if args.max_content_chars < 1:
        parser.error('--max-content-chars must be positive')

    answers = load_optional_mapping(args.answers, list_key='query_id')
    relevance = load_optional_mapping(args.relevance)
    for path in find_run_paths(args.run_path, args.query_id):
        with path.open(encoding='utf-8') as file:
            run = json.load(file)
        print(render_run(run, answers, relevance, args.trajectory, args.max_content_chars))


if __name__ == '__main__':
    main()

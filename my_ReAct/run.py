import argparse
import csv
import json
import os
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()

from local_retrieval.retrieval_client import RetrievalClient
from my_ReAct.prompts import format_research_query
from my_ReAct.runtime import create_llm, create_research_agent
from my_ReAct.settings import AgentSettings, LLMSettings
from my_ReAct.context_manager import BudgetContextManager, RecentTurnsContextManager
from my_ReAct.run_state import RunState
from my_ReAct.experiment import ensure_experiment, fingerprint
from my_ReAct import prompts


def collect_run_data(messages):
    tool_call_counts = {}
    retrieved_docids = []
    seen_docids = set()
    for message in messages:
        for tool_call in message.get('tool_calls', []):
            name = tool_call['name']
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
        if message['role'] != 'tool' or message.get('name') != 'search':
            continue
        observation = message.get('content')
        if not isinstance(observation, dict) or not observation.get('ok'):
            continue
        data = observation.get('data')
        if not isinstance(data, dict):
            continue
        for query_result in data.get('results', []):
            for chunk in query_result.get('chunks', []):
                doc_id = str(chunk['doc_id'])
                if doc_id not in seen_docids:
                    seen_docids.add(doc_id)
                    retrieved_docids.append(doc_id)
    return tool_call_counts, retrieved_docids


def run_query(
    query_id,
    query,
    retriever,
    max_step,
    python_execution_timeout=30.0,
    python_output_limit=20_000,
    llm_settings=None,
    context_turns=None,
    context_chars=None,
    retrieval_info=None,
):
    formatted_query = format_research_query(query)
    agent = None
    answer = None
    error = None
    try:
        llm_settings = llm_settings or LLMSettings.from_env()
        if context_turns is not None and context_chars is not None:
            raise ValueError('context_turns and context_chars are mutually exclusive')
        if context_turns is not None:
            context_manager = RecentTurnsContextManager(context_turns)
        elif context_chars is not None:
            context_manager = BudgetContextManager(max_chars=context_chars)
        else:
            context_manager = None
        agent = create_research_agent(
            query=formatted_query,
            retriever=retriever,
            llm=create_llm(llm_settings),
            settings=AgentSettings(
                max_steps=max_step,
                python_execution_timeout=python_execution_timeout,
                python_output_limit=python_output_limit,
            ),
            context_manager=context_manager,
        )
        agent_result = agent.run()
        answer = agent_result.output
        status = agent_result.status
        termination_reason = agent_result.termination_reason
        if agent_result.error:
            error = f'{agent_result.error.type}: {agent_result.error.message}'
    except Exception as exc:
        status = 'error'
        termination_reason = 'runtime_error'
        error = f'{type(exc).__name__}: {exc}'

    messages = agent.messages if agent else [{'role': 'user', 'content': formatted_query}]
    context_stats = {}
    if agent is not None and getattr(agent, 'context_manager', None) is not None:
        stats = getattr(agent.context_manager, 'stats', {})
        if isinstance(stats, dict):
            context_stats = dict(stats)
    tool_call_counts, retrieved_docids = collect_run_data(messages)
    return {
        'query_id': str(query_id),
        'tool_call_counts': tool_call_counts,
        'status': status,
        'termination_reason': termination_reason,
        'retrieved_docids': retrieved_docids,
        'result': [{'type': 'output_text', 'output': answer or ''}],
        'query': query,
        'model': llm_settings.model if llm_settings else None,
        'api_type': llm_settings.api_type if llm_settings else None,
        'embedder': (retrieval_info or {}).get('embedder'),
        'retrieval': retrieval_info,
        'context': {
            'keep_recent_turns': context_turns,
            'max_chars': context_chars,
            'stats': context_stats,
        },
        'run_state': ({'step': agent.state.step, 'tool_rounds': agent.state.tool_rounds,
                       'elapsed': agent.state.elapsed, 'stop_reason': agent.state.stop_reason}
                      if agent and isinstance(agent.state, RunState) else None),
        'max_step': max_step,
        'usage': [message['usage'] for message in messages if 'usage' in message],
        'error': error,
        'trajectory': messages,
    }


def save_run(record, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{record['query_id']}.json"
    if any(char in str(record['query_id']) for char in '/\\'):
        raise ValueError('query_id cannot contain path separators')
    temporary = output_path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    temporary.replace(output_path)
    return output_path


def load_queries(tsv_path):
    with tsv_path.open(newline='', encoding='utf-8') as file:
        return [(row[0].strip(), row[1].strip()) for row in csv.reader(file, delimiter='\t') if len(row) >= 2]


def completed_query_ids(output_dir):
    completed = set()
    if not output_dir.exists():
        return completed
    for output_path in output_dir.glob('run_*.json'):
        try:
            with output_path.open(encoding='utf-8') as file:
                record = json.load(file)
            if record.get('status') == 'completed':
                completed.add(str(record.get('query_id')))
        except (OSError, json.JSONDecodeError):
            continue
    return completed


def main():

    # ReActAgent → retrieval service (local embedding + FAISS) → continued reasoning

    parser = argparse.ArgumentParser()
    parser.add_argument('query', nargs='?')
    parser.add_argument('--input-tsv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--query-id')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--context-turns', type=int, help='Legacy recent-round window; overrides --context-chars when set')
    parser.add_argument(
        '--context-chars',
        type=int,
        default=120_000,
        help='Maximum visible trajectory characters (default: 120000); full trajectory is still saved',
    )
    parser.add_argument('--max-step', type=int, default=10)
    parser.add_argument('--python-timeout', type=float, default=30.0)
    parser.add_argument('--python-output-limit', type=int, default=20_000)
    parser.add_argument('--retriever-url', default=os.getenv('RETRIEVER_BASE_URL', 'http://127.0.0.1:8002'))
    args = parser.parse_args()
    if not args.query and not args.input_tsv:
        parser.error('provide a query or --input-tsv')
    if args.context_turns is not None and args.context_turns < 1:
        parser.error('--context-turns must be positive')
    if args.context_chars is not None and args.context_chars < 1:
        parser.error('--context-chars must be positive')
    if args.context_turns is not None and args.context_chars is not None:
        # --context-turns is retained as a backwards-compatible explicit
        # override; it disables the default character budget.
        args.context_chars = None
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    AgentSettings(args.max_step, args.python_timeout, args.python_output_limit)

    retriever = RetrievalClient(
        base_url=args.retriever_url,
        api_key=os.getenv('RETRIEVER_API_KEY'),
    )
    output_dir = args.output_dir or Path('runs/hosted_qwen3')

    llm_settings = LLMSettings.from_env()
    retrieval_info = retriever.health()
    if not retrieval_info.get('identity'):
        parser.error('Restart the updated retrieval service: /health must include index identity')
    config = {
        'model': llm_settings.model, 'api_type': llm_settings.api_type,
        'base_url': llm_settings.base_url, 'reasoning_effort': llm_settings.reasoning_effort,
        'retrieval': retrieval_info, 'max_step': args.max_step,
        'python_timeout': args.python_timeout, 'python_output_limit': args.python_output_limit,
        'context_turns': args.context_turns,
        'context_chars': args.context_chars,
        'prompt_hash': fingerprint(Path(prompts.__file__).read_text(encoding='utf-8')),
        'input_hash': fingerprint(args.input_tsv.read_text(encoding='utf-8') if args.input_tsv else args.query),
    }
    experiment_id = ensure_experiment(output_dir, config)

    if args.input_tsv:
        queries = load_queries(args.input_tsv)
        if args.query_id:
            queries = [item for item in queries if item[0] == args.query_id]
        completed = completed_query_ids(output_dir)
        queries = [item for item in queries if item[0] not in completed]
        if args.limit is not None:
            queries = queries[:args.limit]
    else:
        queries = [(args.query_id or 'manual', args.query)]

    for query_id, query in queries:
        record = run_query(
            query_id,
            query,
            retriever,
            args.max_step,
            python_execution_timeout=args.python_timeout,
            python_output_limit=args.python_output_limit,
            llm_settings=llm_settings,
            context_turns=args.context_turns,
            context_chars=args.context_chars,
            retrieval_info=retrieval_info,
        )
        record['experiment_id'] = experiment_id
        output_path = save_run(record, output_dir)
        print(json.dumps({'query_id': query_id, 'status': record['status'], 'output': str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

import json


DEFAULT_TOOL_USE_POLICY = '''Tool-use policy:
- Begin with a batched search of distinct, clue-focused queries when several facts are needed.
- Every later tool call must target a specific unresolved fact.
- Inspect existing snippets before searching again. Do not repeat a semantically equivalent search unless new evidence justifies it.
- Visit a document when its snippet indicates that it may resolve an unresolved fact and the snippet itself is insufficient.
- Once the available evidence supports a unique answer, stop using tools and answer. Do not search merely to increase confidence.
- For multi-clue questions, verify the decisive clues; for simple questions, do not manufacture extra research steps.'''


RESEARCH_QUERY_TEMPLATE = '''You are a deep research agent. Research the question with the fixed search, visit, and python tools as needed, and ground the final answer in retrieved evidence. Search accepts multiple queries in one call and returns top-k chunks. Visit reads cleaned fixed-corpus or static web content and accepts an optional extraction goal. Python can calculate or process data and preserves variables during this task. For questions with multiple clues, verify the decisive clues before answering; a partial match is not sufficient. If evidence conflicts with a candidate, reject it and continue searching. Use python only for actual calculation or data processing.

Question: {question}
Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}'''


def format_research_query(question):
    return RESEARCH_QUERY_TEMPLATE.format(question=question)


def build_agent_system_prompt(tool_schemas):
    return (
        'You are a ReAct research agent with a fixed tool set. '
        'Use search to retrieve relevant chunks, visit to read cleaned corpus or web content, '
        'and python for calculations or data processing. Python variables persist for this task. '
        'For a long document, visit may return a goal-focused extraction and a truncated content '
        'view. Treat that as partial evidence: if another clue is needed, revisit the same '
        'reference with a new goal or a character range; do not conclude that an omitted fact is '
        'absent. Use python only when it processes retrieved facts or performs a real calculation. '
        f'\n\n{DEFAULT_TOOL_USE_POLICY}\n\n'
        'Tool definitions:\n'
        f'{json.dumps(tool_schemas, ensure_ascii=False)}'
    )

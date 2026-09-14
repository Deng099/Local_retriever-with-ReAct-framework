from my_ReAct.base_tool import BaseTool


class SearchTool(BaseTool):
    def __init__(self, retriever):
        super().__init__(
            name='search',
            description='Search the fixed local corpus for the most similar chunks. Supports multiple queries in one call.',
            parameters={
                'type': 'object',
                'properties': {
                    'queries': {
                        'type': 'array',
                        'minItems': 1,
                        'items': {
                            'type': 'object',
                            'properties': {
                                'query': {'type': 'string'},
                                'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 1000, 'default': 5},
                            },
                            'required': ['query'],
                        },
                    },
                },
                'required': ['queries'],
            },
        )
        self.retriever = retriever

    def _execute(self, queries):
        query_texts = [item['query'] for item in queries]
        top_ks = [item.get('top_k', 5) for item in queries]
        batch_results = self.retriever.search_batch(query_texts, top_ks)
        return {
            'results': [
                {
                    'query': query,
                    'chunks': [
                        {
                            'doc_id': chunk['doc_id'],
                            'chunk_id': chunk['chunk_id'],
                            'score': chunk['score'],
                            'snippet': chunk['text'],
                        }
                        for chunk in chunks
                    ],
                }
                for query, chunks in zip(query_texts, batch_results)
            ]
        }

import unittest

from local_retrieval.embedder import Qwen3Embedder


class EmbedderTest(unittest.TestCase):
    def test_query_format_matches_index_build_configuration(self):
        self.assertEqual(
            'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:q',
            Qwen3Embedder.format_query('q'),
        )


if __name__ == '__main__':
    unittest.main()

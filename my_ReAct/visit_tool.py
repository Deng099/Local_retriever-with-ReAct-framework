from my_ReAct.base_tool import BaseTool


class VisitTool(BaseTool):
    def __init__(self, visitor):
        super().__init__(
            name='visit',
            description='Read cleaned content by corpus doc_id or URL. Optionally extract information relevant to a goal.',
            parameters={
                'type': 'object',
                'properties': {
                    'reference': {
                        'type': 'string',
                        'description': 'A doc_id or URL returned by search.',
                    },
                    'goal': {
                        'type': 'string',
                        'description': 'Optional information-extraction goal for this document.',
                    },
                    'offset': {
                        'type': 'integer',
                        'minimum': 0,
                        'default': 0,
                        'description': 'Optional character offset for reading another document range.',
                    },
                    'limit': {
                        'type': 'integer',
                        'minimum': 1,
                        'description': 'Optional maximum characters for a range read.',
                    },
                },
                'required': ['reference'],
            },
        )
        self.visitor = visitor

    def _execute(self, reference, goal=None, offset=0, limit=None):
        if offset or limit is not None:
            return self.visitor.visit(reference, goal, offset=offset, limit=limit)
        return self.visitor.visit(reference, goal)

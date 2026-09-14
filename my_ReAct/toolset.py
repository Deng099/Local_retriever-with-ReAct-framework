from dataclasses import dataclass

from my_ReAct.base_tool import BaseTool


@dataclass(frozen=True)
class ResearchTools:
    search: BaseTool
    visit: BaseTool
    python: BaseTool

    def __post_init__(self):
        names = [tool.name for tool in self.as_list()]
        if names != ['search', 'visit', 'python']:
            raise ValueError(
                f'ResearchTools requires search, visit, python; received {names}'
            )

    def as_list(self):
        return [self.search, self.visit, self.python]

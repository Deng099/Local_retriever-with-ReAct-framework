# basetool(obj)  --> registry --> get_schema() --> llm, 告诉LLM有什么工具
# execute: llm 选择tool, parameters --> RegisterTool.execute

from abc import ABC, abstractmethod

from my_ReAct.contracts import ToolResult

class BaseTool(ABC):
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, *args, **kwargs) -> ToolResult:
        try:
            result = self._execute(*args, **kwargs)
        except Exception as exc:
            return ToolResult.failure(type(exc).__name__, str(exc))
        return result if isinstance(result, ToolResult) else ToolResult.success(result)

    @abstractmethod
    def _execute(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        pass

    # tool object ---> dict, 把tool信息换成llm方便对接的格式
    # tool信息及所需参数
    def to_schema(self) -> dict:   
        return {
            'type': 'function',
            'function':{
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            },
        }


# tool池，统一从这里调用tool: get_schema给llm / execute
class RegisterTool:
    def __init__(self, tools: list[BaseTool]):
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError('Tool names must be unique')

    def get_tool(self, name) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}")

    # tool name(str) --> execute
    def execute(self, name, **kwargs):
        try:
            tool = self.get_tool(name)
        except KeyError as exc:
            return ToolResult.failure('UnknownTool', exc.args[0]).to_dict()
        return tool.execute(**kwargs).to_dict()

    def get_tool_schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self._tools.values()]

    def close_all(self):
        errors = []
        for tool in reversed(list(self._tools.values())):
            try:
                tool.close()
            except Exception as exc:
                errors.append({
                    'tool': tool.name,
                    'type': type(exc).__name__,
                    'message': str(exc),
                })
        return errors
    


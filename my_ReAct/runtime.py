from my_ReAct.openai_llm import OpenAICompatibleChatLLM, OpenAIResponsesLLM
from my_ReAct.python_tool import PythonTool
from my_ReAct.react import ReActAgent
from my_ReAct.search_tool import SearchTool
from my_ReAct.settings import AgentSettings
from my_ReAct.toolset import ResearchTools
from my_ReAct.visit_tool import VisitTool


def create_llm(settings):
    if settings.api_type == 'chat':
        return OpenAICompatibleChatLLM(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            trust_env=settings.trust_env,
        )
    if settings.api_type == 'responses':
        return OpenAIResponsesLLM(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            reasoning_effort=settings.reasoning_effort,
            trust_env=settings.trust_env,
        )
    raise ValueError(f'Unsupported API type: {settings.api_type}')


def create_research_tools(retriever, settings):
    return ResearchTools(
        search=SearchTool(retriever),
        visit=VisitTool(retriever),
        python=PythonTool(
            execution_timeout=settings.python_execution_timeout,
            output_limit=settings.python_output_limit,
        ),
    )


def create_research_agent(query, retriever, llm, settings=None, context_manager=None):
    settings = settings or AgentSettings()
    return ReActAgent(
        llm=llm,
        max_step=settings.max_steps,
        query=query,
        tools=create_research_tools(retriever, settings),
        context_manager=context_manager,
    )

from langgraph.config import get_stream_writer
from langgraph.types import Command

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.agent.code_analysis_agent import CodeAnalysisAgent
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update

from src.llm import LLMFactory, LLMType
from src.utils.embedding_util import EmbeddingUtil



class ModelAgent:
    def __init__(self):
        self.agent = CodeAnalysisAgent(chat_model=LLMFactory.create_llm(LLMType.DEEPSEEK_CHAT),reason_model=LLMFactory.create_llm(LLMType.DEEPSEEK_REASON),embedding_model=EmbeddingUtil())

    async def __call__(self, state: AdapterState) -> Command:
        try:
            path = state['model_path']
        except KeyError:
            get_stream_writer()({
                "data": AiChatResultVO(text="❌ No model path provided").model_dump_json(
                    exclude_none=True
                )
            })
            return Command(
                goto=AgentTypeEnum.Supervisor.value,
                update=await command_update(state),
            )

        prompt = state['model_agent_prompt'][-1]
        output = None
        for i in range(5):
            output = self.agent.action(path, prompt)
            if output['success']:
                break
            else:
                pass
        if output['success']:
            state['model_analyse'].append({'markdown':output['markdown'],"json_out":output['json_out'],
                    "summary":output['summary']})
            get_stream_writer()({
                "data": AiChatResultVO(text="✅resolve model successful").model_dump_json(
                            exclude_none=True
                        )
            })
        else:
            get_stream_writer()({
                        "data": AiChatResultVO(text="Can't resolve model").model_dump_json(
                            exclude_none=True
                        )
                    })
        return Command(
            goto=AgentTypeEnum.Supervisor.value,
            update=await command_update(state),
        )

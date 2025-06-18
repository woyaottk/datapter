import logging

from langgraph.types import Command

from src.domain.agent.code_analysis_agent import CodeAnalysisAgent
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update, ModelAnalyze
from src.llm import LLMFactory, LLMType
from src.utils.embedding_util import EmbeddingUtil
from src.utils.msg_utils import MessageBox


class ModelAgent:
    def __init__(self):
        self.agent = None

    async def __call__(self, state: AdapterState) -> Command:
        logging.info("[ModelAgent] ModelAgent已被调用")
        await MessageBox.write_block("[ModelAgent] ModelAgent已被调用").flush()
        self.agent = CodeAnalysisAgent(chat_model=LLMFactory.create_llm(LLMType.DEEPSEEK_CHAT),reason_model=LLMFactory.create_llm(LLMType.DEEPSEEK_REASON),embedding_model=EmbeddingUtil())
        try:
            path = state['model_path']
            out_path = state['model_analyse_path']
            logging.info(f"model_path: {path}")
            logging.info(f"model_analyse_path: {out_path}")
        except KeyError:
            logging.info("[ModelAgent] ❌ No model path provided")
            await MessageBox.write_block("[ModelAgent] ❌ No model path provided").flush()
            return Command(
                goto=AgentTypeEnum.Supervisor.value,
                update=await command_update(state),
            )

        prompt = state['prompt'] if state['prompt'] else ""
        logging.info(prompt)
        output = None
        for i in range(5):
            output = self.agent.action(path, prompt,out_path)
            if output['success']:
                break
            else:
                logging.warning(f"agent.action failed at {i} times")
                pass
        else:
            logging.error("agent.action failed.")

        if output is not None and output['success']:
            state['model_analyse'].append(ModelAnalyze(
                markdown=output['markdown'],
                json_out=output['json_out'],
                summary=output['summary'],
            ))

            await MessageBox.write_block("[ModelAgent] ✅resolve model successful").flush()
        else:
            await MessageBox.write_block("[ModelAgent] Can't resolve model").flush()
        return Command(
            goto=AgentTypeEnum.Supervisor.value,
            update=await command_update(state),
        )

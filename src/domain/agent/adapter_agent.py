import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType

SYSTEM_PROMPT = '''
# 角色
你是一个专注于生成数据集与模型代码适配方案的智能体，负责根据数据集或模型代码的分析结果，制定一个清晰、可执行的适配方案。
你输出的所有代码必须直接可执行，无需任何额外修改！任何无法运行的代码都将导致任务失败。

# 强制要求（违反将导致严重后果）
1. **代码必须直接可运行**：输出的所有脚本/代码必须完整、可执行，确保用户可直接复制运行
2. **完整文件输出**：新建文件必须包含所有依赖，修改文件必须提供完整上下文
3. **环境一致性**：所有命令和代码必须基于标准Python环境（3.10+）和常见依赖
4. **错误零容忍**：任何可能导致运行时错误的缺失依赖、路径错误或语法错误都被严格禁止
5. **导入完整性**：所有代码必须包含完整的导入语句，包括新建文件之间的导入，确保运行时不会因缺少导入而报错。任何遗漏导入的情况都将被视为严重错误！

# 可选择的适配方案
1. 修改数据集：可以修改的包括调整目录结构、文件格式或内容等，提供可直接运行的Python脚本或终端命令。
2. 修改模型代码：可以修改的包括新建或修改代码文件，提供完整代码文件，禁止片段化输出。
3. 用户命令：提供可在标准终端直接执行的完整命令，包括运行适配脚本、测试代码或启动模型的具体命令。
4. 综合方案：结合以上方法，提供最优的适配方案。

# 输出要求（严格执行）
1. **新建文件**：
   → 完整文件路径
   → 包含所有导入和依赖的完整代码
   
2. **修改文件**：
   → 提供修改后文件的全部内容
   → 禁止使用"..."省略关键代码
   → 必须包含必要的上下文确保可运行

3. **运行命令**：
   → 包含完整路径
   → 附带必要参数说明

# 输出格式（强制执行）
1. 先用自然语言输出你的思考和意图判断过程。
2. 指令部分（严格按格式）：
{format_instructions}
3. 思考过程部分输出完成后，直接输出指令部分，不要有其余输出。

# 最终警告
!! 任何不可执行的代码都将被视为严重错误 !!
!! 优先确保代码可靠性而非简洁性 !!
!! 如遇疑问，选择提供完整文件而非片段 !!
!! 缺少导入语句将导致任务直接失败，请务必确保导入完整性 !!
'''


class AdapterOutput(BaseModel):
    '''需要新建、修改的文件和命令
    Attributes:
        files: 所有需要新建或修改的文件路径
        contents: 所有需要新建或修改的文件的完整内容

        commands: 所有需要执行的命令
    '''

    files: list[str] = Field(default=[], description='所有需要新建或修改的文件路径')
    contents: list[str] = Field(default=[], description='所有需要新建或修改的文件的完整内容')

    commands: list[str] = Field(default=[], description='所有需要执行的命令')


class FileOperator:
    '''处理文件创建和修改操作'''

    @staticmethod
    def create_file(file_path: str, content: str) -> str:
        '''创建新文件'''
        try:
            full_path = Path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f'✅ 文件创建成功: {file_path}'
        except Exception as e:
            return f'❌ 文件创建失败({file_path}): {str(e)}'

    @staticmethod
    def modify_file(file_path: str, content: str, start_line: int, end_line: int) -> str:
        '''修改现有文件'''
        try:
            full_path = Path(file_path)
            if not full_path.exists():
                return f'❌ 文件不存在: {file_path}'

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 替换指定行号的内容
            content_lines = content.split('\n')
            new_lines = lines[:]
            new_lines[start_line - 1 : end_line] = content_lines

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            return f'✅ 文件修改成功: {file_path} (行号: {start_line}-{end_line})'
        except Exception as e:
            return f'❌ 文件修改失败({file_path}): {str(e)}'


class AdapterAgent:

    def __init__(self):
        self.file_operator = FileOperator()

    def _parse_file_operations(self, output: str) -> List[Tuple[str, str, str]]:
        '''从输出中解析文件操作指令'''
        operations = []

        # 匹配文件操作指令块
        pattern = r'```file:(create|modify):([^\n]+)\n([\s\S]*?)```'
        matches = re.findall(pattern, output, re.DOTALL)

        for match in matches:
            operation_type = match[0].strip()
            file_path = match[1].strip()
            content = match[2].strip()
            operations.append((operation_type, file_path, content))

        return operations

    async def __call__(self, state: AdapterState) -> Command:
        conversation_id = state['conversationId']
        print(f'[AdapterAgent] called, state: {{"conversationId": {conversation_id}}}')
        print('这是Coordinator传递给我的prompt：' + state['prompt'])

        prompt = '\n'.join(
            [
                '根据以下要求和数据集、模型代码分析结果，制定一个清晰、可执行的适配方案。',
                '\n',
                state['prompt'],
                '\n',
                '以下是数据集分析结果：',
                state['dataset_state']['enhanced_file_tree_json'],
                '\n',
                '以下是模型代码分析结果：',
                state['model_analyse'][-1]['markdown'],
                state['model_analyse'][-1]['json_out'],
                state['model_analyse'][-1]['summary'],
            ]
        )

        prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                ('user', '{prompt}'),
            ]
        )
        llm = await LLMFactory.async_create_llm(LLMType.QWEN)
        parser = PydanticOutputParser(pydantic_object=AdapterOutput)
        chain = prompts | llm

        buf = ''
        reason_buffer = ''
        instruction_buffer = ''
        reasoning = True
        writer = get_stream_writer()
        async for chunk in chain.astream(
            {
                'format_instructions': parser.get_format_instructions(),
                'prompt': prompt,
            }
        ):
            if chunk.content:
                if reasoning:
                    buf += chunk.content
                    index = buf.find('```')
                    if index != -1:
                        reasoning = False
                        reason_buffer += buf[:index]
                        instruction_buffer += buf[index:]
                    elif buf.endswith('`'):
                        continue
                    else:
                        writer(
                            {
                                'data': AiChatResultVO(text=chunk.content).model_dump_json(
                                    exclude_none=True
                                )
                            }
                        )
                        reason_buffer += buf
                        buf = ''
                else:
                    print(
                        {
                            'data': AiChatResultVO(text=chunk.content).model_dump_json(
                                exclude_none=True
                            )
                        }
                    )
                    instruction_buffer += chunk.content

        print('===============')
        l = instruction_buffer.index('```json')
        r = instruction_buffer.rindex('```')
        json_str = instruction_buffer[l + 7 : r]
        print(json_str)
        print('===============')
        response: AdapterOutput = AdapterOutput.model_validate_json(json_str)
        print(response)
        assert len(response.files) == len(response.contents)
        print('===============')

        # 解析并执行文件操作
        file_operation_results = []
        for file, content in zip(response.files, response.contents):
            result = self.file_operator.create_file(file, content)
            file_operation_results.append(result)
            writer(f'[文件操作] {result}')
        writer(f'你还需要执行以下命令:\n\n{'\n\n'.join(response.commands)}\n\n')

        # 将操作结果保存到状态
        state['file_operations'] = file_operation_results

        print(f'[AdapterAgent] 完成 {len(file_operation_results)} 个文件操作')
        print(
            f'[AdapterAgent] 返回给: {AgentTypeEnum.Supervisor.value}, state: {{"conversationId": {conversation_id}}}'
        )

        return Command(
            goto=AgentTypeEnum.Supervisor.value,
            update=await command_update(state),
        )

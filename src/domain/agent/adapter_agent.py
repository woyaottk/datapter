import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.config import get_stream_writer
from langgraph.types import Command

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType

SYSTEM_PROMPT = '''
你是一个专注于数据集与模型代码适配的智能体。
用户提供了一对数据集与模型代码，目标是让数据集能够正确运行在模型代码上。
你需要根据提供的分析结果，生成一个详细且可执行的适配方案。

## 文件操作格式要求
当需要创建或修改文件时，请使用以下格式：

### 创建新文件
```file:create:path/to/new_file.py
# 文件内容...
# 包含完整的代码和注释
```

### 修改现有文件
```file:modify:path/to/existing_file.py
# 起始行号
# 要插入的代码内容...
# 结束行号
```

## 适配方案内容要求
1. **数据集调整**：
   - 提供完整的 Python 脚本，确保可以直接运行
   - 脚本需包含清晰的注释

2. **模型代码调整**：
   - 明确指出需要修改的文件路径和行号
   - 提供修改后的完整代码片段

3. **综合调整**：
   - 结合数据集和模型代码进行优化

确保：
- 所有脚本和代码修改都是完整且可执行的
- 方案逻辑清晰，步骤明确
- 避免冗长描述，聚焦解决问题
'''


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
    def modify_file(file_path: str, content: str, start_line: Optional[int] = None) -> str:
        '''修改现有文件'''
        try:
            full_path = Path(file_path)
            if not full_path.exists():
                return f'❌ 文件不存在: {file_path}'

            # 解析修改内容中的行号标记
            if start_line is None:
                # 尝试从内容中提取行号
                match = re.search(r'#\s*起始行号:\s*(\d+)', content)
                if match:
                    start_line = int(match.group(1))
                    content = content.replace(match.group(0), '')

            if start_line is None:
                return f'❌ 未指定修改行号: {file_path}'

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 替换指定行号的内容
            content_lines = content.split('\n')
            new_lines = (
                lines[: start_line - 1]
                + content_lines
                + lines[start_line + len(content_lines) - 1 :]
            )

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            return f'✅ 文件修改成功: {file_path} (行号: {start_line}-{start_line+len(content_lines)-1})'
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
        print('这是Coordinator传递给我的context：' + state['context'])

        user_question = state['context']

        prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                ('user', '{user_question}'),
            ]
        )
        llm = await LLMFactory.async_create_llm(LLMType.QWEN)
        chain = prompts | llm

        writer = get_stream_writer()
        # 清空上下文并准备接收新内容
        state['context'] = ''
        file_operation_results = []

        async for chunk in chain.astream({'user_question': user_question}):
            if chunk.content:
                state['context'] += chunk.content
                # 这里保留原有的流式输出逻辑
                writer(
                    {'data': AiChatResultVO(text=chunk.content).model_dump_json(exclude_none=True)}
                )

        # 解析并执行文件操作
        operations = self._parse_file_operations(state['context'])
        for op_type, file_path, content in operations:
            if op_type == 'create':
                result = self.file_operator.create_file(file_path, content)
            elif op_type == 'modify':
                result = self.file_operator.modify_file(file_path, content)
            else:
                result = f'❌ 未知操作类型: {op_type}'

            file_operation_results.append(result)
            print(f'[文件操作] {result}')

        # 将操作结果保存到状态
        state['file_operations'] = file_operation_results

        print(f'[AdapterAgent] 完成 {len(operations)} 个文件操作')
        print(
            f'[AdapterAgent] 返回给: {AgentTypeEnum.Supervisor.value}, state: {{"conversationId": {conversation_id}}}'
        )
        print('这是我传递给Coordinator的context：' + state['context'])

        return Command(
            goto=AgentTypeEnum.Supervisor.value,
            update=await command_update(state),
        )

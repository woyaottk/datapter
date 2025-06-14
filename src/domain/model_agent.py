import os

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.config import get_stream_writer
from langgraph.types import Command

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update

import datetime
import json
from typing import Dict, List, Any


# 导入所需模块
# from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.file_management import ListDirectoryTool, FileSearchTool, ReadFileTool,WriteFileTool
from langgraph.graph import StateGraph, MessagesState, START, END


# RAG系统相关导入
from langchain_chroma import Chroma
# from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from src.llm import LLMFactory, LLMType
from src.utils.embedding_util import EmbeddingUtil

# 定义状态类
class CollaborativeAgentState(MessagesState):
    """协作代理系统的状态类"""
    information_summary: str = ""
    need_more_info: bool = True
    current_working_path: str = "./"  # 改为通用的默认路径
    discovered_paths: List[str] = []  # 添加已发现路径列表
    path_context: Dict[str, Any] = {}  # 添加路径上下文信息
    target_base_path: str = "./"  # 添加目标基础路径，用于存储用户指定的根路径

# tool工具模型节点
def tool_tool_node(state):
    """使用tool模型调用工具获取信息"""
    messages = state["messages"]
    current_path = state.get("current_working_path", "./")
    discovered_paths = state.get("discovered_paths", [])
    path_context = state.get("path_context", {})
    target_base_path = state.get("target_base_path", "./")
    
    try:
        # 在调用工具前，为模型提供当前路径上下文
        if messages and messages[-1].type == "human":
            # 检查消息中是否包含路径操作
            last_content = messages[-1].content.lower()
            
            # 动态构建项目结构建议
            project_structure_hints = []
            if discovered_paths:
                # 基于已发现路径推断常见目录模式
                for path in discovered_paths[-5:]:
                    if "data" in path:
                        project_structure_hints.append(f"{path} (data)")
                    elif "model" in path:
                        project_structure_hints.append(f"{path} (models)")
                    elif "task" in path:
                        project_structure_hints.append(f"{path} (tasks)")
                    elif "example" in path:
                        project_structure_hints.append(f"{path} (examples)")
            
            if not project_structure_hints:
                # 如果没有发现路径，提供通用建议
                project_structure_hints = [
                    f"{target_base_path}data/ (potential data directory)",
                    f"{target_base_path}src/ (potential source directory)", 
                    f"{target_base_path}models/ (potential models directory)",
                    f"{target_base_path}examples/ (potential examples directory)"
                ]
            
            # 构建路径上下文提示
            path_context_msg = f"""
            CRITICAL PATH CONTEXT - MUST FOLLOW:
            - Base Target Path: {target_base_path}
            - Current Working Path: {current_path}
            - Discovered Paths: {discovered_paths[-5:] if discovered_paths else ['None yet']}
            - Suggested Project Structure: {project_structure_hints[:4]}
            
            🔥 MANDATORY PATH USAGE RULES:
            1. NEVER use relative paths - ALWAYS use full paths starting with {target_base_path}
            2. Tool call examples:
               ✅ CORRECT: ListDirectoryTool("{target_base_path}fairseq/")
               ✅ CORRECT: ReadFileTool("{target_base_path}fairseq/data/data_utils.py")
               ❌ WRONG: ListDirectoryTool("fairseq/")
               ❌ WRONG: ReadFileTool("fairseq/data/data_utils.py")
            3. If path fails, try these alternatives with FULL PATHS:
               - {target_base_path}
               - {target_base_path}src/
               - {target_base_path}data/
               - {target_base_path}examples/
               - {target_base_path}lib/
            4. Use FileSearchTool to locate files if unsure of exact path
            5. Build on successfully discovered paths: {discovered_paths[-3:] if discovered_paths else ['None']}
            
            REMEMBER: Every tool call must start with {target_base_path} - this is CRITICAL!
            
            USER REQUEST: {messages[-1].content}
            """
            
            # 修改最后一条消息以包含路径上下文
            enhanced_message = HumanMessage(content=path_context_msg)
            messages = messages[:-1] + [enhanced_message]
        
        get_stream_writer()({
            "data": AiChatResultVO(text="🔧 开始调用工具模型进行代码分析...").model_dump_json(
                exclude_none=True
            )
        })
        print(f"🔧 [TOOL_NODE] 输入消息数量: {len(messages)}")
        if messages:
            print(f"🔧 [TOOL_NODE] 最后一条消息类型: {messages[-1].type}")
            print(f"🔧 [TOOL_NODE] 最后一条消息预览: {messages[-1].content[:150]}...")
        
        result = tool_agent.invoke({"messages": messages}, {"recursion_limit": 200})
        
        # 详细显示工具模型的输出
        print("\n" + "="*60)
        print("🔧 [TOOL_MODEL_OUTPUT] 工具模型完整输出:")
        print("="*60)
        
        if result.get("messages"):
            get_stream_writer()({
                "data": AiChatResultVO(text=f"📊 工具模型返回了 {len(result['messages'])} 条消息").model_dump_json(
                    exclude_none=True
                )
            })
            print(f"📊 返回消息数量: {len(result['messages'])}")
            for i, msg in enumerate(result["messages"]):
                if msg.type == "ai" and len(msg.content) > 100:  # 只输出重要的AI回复
                    get_stream_writer()({
                        "data": AiChatResultVO(text=f"🔧 工具分析结果 {i+1}: {msg.content[:200]}...").model_dump_json(
                            exclude_none=True
                        )
                    })
                print(f"\n--- 消息 {i+1} ---")
                print(f"类型: {msg.type}")
                print(f"内容长度: {len(msg.content)} 字符")
                print("内容:")
                print(msg.content)
                print("-" * 40)
        else:
            get_stream_writer()({
                "data": AiChatResultVO(text="⚠️ 工具模型没有返回任何消息").model_dump_json(
                    exclude_none=True
                )
            })
            print("⚠️ 工具模型没有返回任何消息")
        
        print("="*60)
        print("🔧 [TOOL_MODEL_OUTPUT] 输出结束")
        print("="*60 + "\n")
        
        # 更新路径状态
        updated_path = current_path
        updated_discovered = discovered_paths.copy()
        updated_context = path_context.copy()
        
        # 检查是否成功获取了工具结果
        if result.get("messages"):
            last_message = result["messages"][-1]
            
            # 从工具响应中提取路径信息
            if hasattr(last_message, 'content'):
                content = last_message.content
                
                # 提取成功访问的路径
                import re
                successful_paths = re.findall(r'(?:Listed|Reading|Found).*?([./\w-]+/[./\w-]+)', content)
                for path in successful_paths:
                    if path not in updated_discovered:
                        updated_discovered.append(path)
                        get_stream_writer()({
                            "data": AiChatResultVO(text=f"🔍 发现新路径: {path}").model_dump_json(
                                exclude_none=True
                            )
                        })
                        print(f"Discovered new path: {path}")
                
                # 检测路径错误并提供智能恢复
                error_patterns = [
                    ("no such file or directory", "PATH_ERROR"),
                    ("Error:", "GENERAL_ERROR"),
                    ("could not", "ACCESS_ERROR"),
                    ("permission denied", "PERMISSION_ERROR")
                ]
                
                # 特别检测ReadFileTool错误
                readfile_error = False
                if "no such file or directory" in content.lower() and "readfiletool" in content.lower():
                    readfile_error = True
                    print("Detected ReadFileTool path error - suggesting FileSearchTool workflow")
                
                error_type = None
                for pattern, err_type in error_patterns:
                    if pattern.lower() in content.lower():
                        error_type = err_type
                        break
                
                # 如果检测到ReadFileTool特定错误，提供专门的解决方案
                if readfile_error:
                    recovery_message = HumanMessage(content=f"""
                    READFILETOOL ERROR DETECTED! Here's the correct workflow:
                    
                    🔥 CRITICAL: ReadFileTool failed because you used an incorrect path.
                    
                    CORRECT READFILETOOL WORKFLOW:
                    1. NEVER guess file paths for ReadFileTool
                    2. ALWAYS use this 2-step process:
                       
                       Step 1: FileSearchTool("filename.py") to find exact location
                       Step 2: ReadFileTool("{target_base_path}" + exact_path_from_step1)
                    
                    EXAMPLE:
                    - FileSearchTool("data_utils.py") returns "fairseq/fairseq/data/data_utils.py"
                    - Then use ReadFileTool("{target_base_path}fairseq/fairseq/data/data_utils.py")
                    
                    PROJECT CONTEXT:
                    - Base Path: {target_base_path}
                    - Many projects have nested directory structures
                    - FileSearchTool reveals the exact nested path structure
                    
                    Try the FileSearchTool + ReadFileTool workflow now!
                    """)
                    
                    result["messages"].append(recovery_message)
                    
                # 如果检测到路径错误，提供智能路径建议
                elif error_type == "PATH_ERROR":
                    print(f"Detected {error_type}: {content[:200]}")
                    
                    # 动态构建路径建议
                    suggested_paths = [
                        target_base_path,
                        f"{target_base_path}src/",
                        f"{target_base_path}data/", 
                        f"{target_base_path}models/",
                        f"{target_base_path}examples/",
                        f"{target_base_path}lib/",
                        f"{target_base_path}tasks/",
                        f"{target_base_path}utils/"
                    ]
                    
                    # 如果有历史成功路径，优先推荐相似路径
                    if updated_discovered:
                        # 分析成功路径的模式
                        for successful_path in updated_discovered[-3:]:
                            parent_dir = "/".join(successful_path.split("/")[:-1]) + "/"
                            if parent_dir not in suggested_paths:
                                suggested_paths.insert(0, parent_dir)
                        
                        # 基于成功路径推断兄弟目录
                        last_successful = updated_discovered[-1]
                        path_parts = last_successful.split("/")
                        if len(path_parts) > 2:
                            base_path = "/".join(path_parts[:-2]) + "/"
                            suggested_paths.insert(0, base_path)
                    
                    recovery_message = HumanMessage(content=f"""
                    Path access failed. Let me use systematic path exploration with FULL PATHS:
                    
                    🔥 CRITICAL: Use FULL paths starting with {target_base_path}
                    
                    IMMEDIATE ACTIONS:
                    1. Try these specific FULL paths in order:
                       {chr(10).join(f'   - ListDirectoryTool("{path}")' for path in suggested_paths[:4])}
                    
                    2. If still failing, use FileSearchTool("*.py") to find files
                    
                    3. For ReadFileTool, ALWAYS:
                       a) Use FileSearchTool("filename.py") first to get exact path
                       b) Then use ReadFileTool("{target_base_path}" + exact_path_from_search)
                    
                    PROJECT CONTEXT:
                    - Base Path: {target_base_path}
                    - Current Path: {current_path}
                    - Last Successful: {updated_discovered[-3:] if updated_discovered else 'None'}
                    
                    REMEMBER: NEVER use relative paths like "fairseq/data/" - ALWAYS use "{target_base_path}fairseq/data/"
                    
                    Start with the first suggested path above.
                    """)
                    
                    result["messages"].append(recovery_message)
                    
                elif error_type in ["ACCESS_ERROR", "GENERAL_ERROR"]:
                    recovery_message = HumanMessage(content=f"""
                    Access error encountered. Using alternative approach with FULL PATHS:
                    
                    🔥 CRITICAL: Always use FULL paths starting with {target_base_path}
                    
                    1. FileSearchTool("dataset.py") - locate dataset files
                    2. FileSearchTool("data") - find data directories  
                    3. ListDirectoryTool("{target_base_path}") - explore base directory
                    4. Check these common project locations with FULL PATHS:
                       - ListDirectoryTool("{target_base_path}src/")
                       - ListDirectoryTool("{target_base_path}data/")
                       - ListDirectoryTool("{target_base_path}examples/")
                    5. For reading files: 
                       a) FileSearchTool("filename.py") to get exact path
                       b) ReadFileTool("{target_base_path}" + exact_path_from_search)
                    
                    Base path: {target_base_path}
                    Current context: {current_path}
                    
                    REMEMBER: NEVER use paths like "src/" - ALWAYS use "{target_base_path}src/"
                    """)
                    result["messages"].append(recovery_message)
        
        return {
            "messages": result["messages"],
            "information_summary": state.get("information_summary", ""),
            "need_more_info": True,
            "current_working_path": updated_path,
            "discovered_paths": updated_discovered,
            "path_context": updated_context,
            "target_base_path": target_base_path
            }
    except Exception as e:
        print(f"Tool node execution error: {e}")
        error_message = AIMessage(content=f"Tool call encountered an error: {str(e)}. Let me try a more systematic approach to explore the codebase.")
        
        # 提供基于当前路径状态的恢复建议
        recovery_message = HumanMessage(content=f"""
        System error occurred. Using systematic recovery with FULL PATHS:
        
        🔥 CRITICAL: Always use FULL paths starting with {target_base_path}
        
        1. Start with: ListDirectoryTool("{target_base_path}")
        2. Then try: ListDirectoryTool("{current_path}")
        3. Use FileSearchTool to locate target files in base directory
        4. Explore common project directories systematically:
           - ListDirectoryTool("{target_base_path}src/")
           - ListDirectoryTool("{target_base_path}data/")
           - ListDirectoryTool("{target_base_path}examples/")
        5. For file reading: 
           a) FileSearchTool("filename.py") to get exact path
           b) ReadFileTool("{target_base_path}" + exact_path_from_search)
        
        Base path: {target_base_path}
        Current working path: {current_path}
        Known paths: {discovered_paths[-3:] if discovered_paths else 'None'}
        
        REMEMBER: Every tool call must include the full base path {target_base_path}
        """)
        
        return {
            "messages": state["messages"] + [error_message, recovery_message],
            "information_summary": state.get("information_summary", ""),
            "need_more_info": True,
            "current_working_path": current_path,
            "discovered_paths": discovered_paths,
            "path_context": path_context,
            "target_base_path": target_base_path
    }

# DeepSeek推理模型节点
def reason_node(state):
    """使用DeepSeek推理模型分析信息并决定是否需要更多信息"""
    messages = state["messages"]
    current_summary = state.get("information_summary", "")
    current_path = state.get("current_working_path", "./")
    discovered_paths = state.get("discovered_paths", [])
    path_context = state.get("path_context", {})
    target_base_path = state.get("target_base_path", "./")
    
    # 动态分析项目结构
    project_structure_analysis = "Unknown structure"
    if discovered_paths:
        structure_patterns = []
        for path in discovered_paths:
            if "data" in path.lower():
                structure_patterns.append("data-focused")
            if "model" in path.lower():
                structure_patterns.append("model-centric")
            if "src" in path.lower():
                structure_patterns.append("source-organized")
            if "example" in path.lower():
                structure_patterns.append("example-driven")
        project_structure_analysis = ", ".join(set(structure_patterns)) or "general project"
    
    # 构建系统提示词，包含路径上下文
    system_prompt = f"""
    You are a code analysis assistant. Analyze the collected information and make a CLEAR decision.
    
    FOCUS on these THREE aspects:
    1. Data structure & processing pipeline
    2. Dataloader index organization  
    3. Data augmentation techniques
    
    PROJECT CONTEXT:
    - Base Path: {target_base_path}
    - Current Working Path: {current_path}
    - Successfully Discovered Paths: {discovered_paths[-5:] if discovered_paths else ['None']}
    - Project Structure Type: {project_structure_analysis}
    
    Current Information: {current_summary}
    Latest Info: {messages[-1].content}
    
    DECISION RULES:
    - If you have found key dataset/dataloader files and read their content → NEED_MORE_INFO: false
    - If you have code examples showing data loading/processing → NEED_MORE_INFO: false  
    - If tool encountered repeated file access errors → provide specific alternative paths
    - If no dataset-related files found yet → NEED_MORE_INFO: true
    
    PATH-AWARE INSTRUCTIONS:
    When providing instructions, consider the project context:
    - Use discovered paths: {discovered_paths[-3:] if discovered_paths else [target_base_path]}
    - Focus on project's identified structure type: {project_structure_analysis}
    - Suggest file searches in successful directories
    - Build on successful path patterns
    
    RESPONSE FORMAT:
    ANALYSIS: [Brief analysis of what was found]
    NEED_MORE_INFO: [true/false - be decisive]
    PATH_UPDATE: [Suggest new working path if needed, or "None"]
    INSTRUCTIONS: [If true, give ONE specific instruction with full path like "Read file {target_base_path}src/dataset.py"]
    FINAL_ANSWER: [If false, provide comprehensive answer with code details]
    
    Be DECISIVE and SPECIFIC. Use full paths in instructions based on the project structure.
    """
    
    get_stream_writer()({
        "data": AiChatResultVO(text="🧠 开始推理分析，整理代码分析结果...").model_dump_json(
            exclude_none=True
        )
    })
    print(f"🧠 [REASON_NODE] 输入提示词长度: {len(system_prompt)} 字符")
    print(f"🧠 [REASON_NODE] 当前信息摘要长度: {len(current_summary)} 字符")
    print(f"🧠 [REASON_NODE] 发现的路径数量: {len(discovered_paths)}")
    
    # 调用推理模型
    reasoning_result = reason_model.invoke(system_prompt)
    
    # 详细显示推理模型的输出
    get_stream_writer()({
        "data": AiChatResultVO(text=f"🧠 推理模型分析完成，生成了 {len(reasoning_result.content)} 字符的分析结果").model_dump_json(
            exclude_none=True
        )
    })
    print("\n" + "="*60)
    print("🧠 [REASONING_MODEL_OUTPUT] 推理模型完整输出:")
    print("="*60)
    print(f"📊 输出类型: {type(reasoning_result)}")
    print(f"📊 输出内容长度: {len(reasoning_result.content)} 字符")
    print("\n--- 推理模型完整分析 ---")
    print(reasoning_result.content)
    print("-" * 60)
    print("="*60)
    print("🧠 [REASONING_MODEL_OUTPUT] 输出结束")
    print("="*60 + "\n")
    
    # 解析结果
    analysis = reasoning_result.content
    
    # 提取是否需要更多信息的决定
    need_more = "NEED_MORE_INFO: true" in analysis
    
    # 提取路径更新建议
    updated_path = current_path
    if "PATH_UPDATE:" in analysis:
        path_suggestion = analysis.split("PATH_UPDATE:")[1].split("INSTRUCTIONS:")[0].split("FINAL_ANSWER:")[0].strip()
        if path_suggestion and path_suggestion != "None" and path_suggestion != "":
            updated_path = path_suggestion
            get_stream_writer()({
            "data": AiChatResultVO(text=f"📁 更新工作路径: {current_path} -> {updated_path}").model_dump_json(
                exclude_none=True
            )
        })
        print(f"Updating working path: {current_path} -> {updated_path}")
    
    # 提取最终分析结果
    if "FINAL_ANSWER:" in analysis:
        final_answer = analysis.split("FINAL_ANSWER:")[1].strip()
        messages.append(AIMessage(content=final_answer))
    
    # 如果需要更多信息，提取指令
    if need_more and "INSTRUCTIONS:" in analysis:
        instructions_text = analysis.split("INSTRUCTIONS:")[1].split("FINAL_ANSWER:")[0].strip()
        
        # 增强指令，确保包含路径上下文
        enhanced_instructions = f"""
        {instructions_text}
        
        PATH CONTEXT REMINDER:
        - Base Path: {target_base_path}
        - Current Path: {updated_path}
        - Known Good Paths: {discovered_paths[-3:] if discovered_paths else ['None yet']}
        - Project Type: {project_structure_analysis}
        - If path fails, try similar patterns from successful paths
        """
        
        messages.append(HumanMessage(content=enhanced_instructions))
    
    # 更新信息摘要
    if "ANALYSIS:" in analysis:
        new_analysis = analysis.split("ANALYSIS:")[1].split("NEED_MORE_INFO:")[0].strip()
        updated_summary = current_summary + "\n\nNew Analysis: " + new_analysis
    else:
        updated_summary = current_summary
    
    return {
        "messages": messages,
        "information_summary": updated_summary,
        "need_more_info": need_more,
        "current_working_path": updated_path,
        "discovered_paths": discovered_paths,
        "path_context": path_context,
        "target_base_path": target_base_path
    }

# 决策路由节点
def router(state):
    """根据状态决定下一步操作"""
    if not state["need_more_info"]:
        return END
    return "reason_node" if state["messages"][-1].type == "ai" else "tool_tool_node"

def create_workflow(memory):
    """创建工作流图"""
    workflow = StateGraph(CollaborativeAgentState)
    workflow.add_node("tool_tool_node", tool_tool_node)
    workflow.add_node("reason_node", reason_node)

    # 定义边缘关系
    workflow.add_conditional_edges(START, router)
    workflow.add_conditional_edges("tool_tool_node", router)
    workflow.add_conditional_edges("reason_node", router)

    # 编译图时传入memory作为checkpointer
    return workflow.compile(checkpointer=memory)

def extract_dataset_info_from_analysis(messages, summary):
    """从分析结果中提取数据集相关信息 - 基于代码的详细版本"""
    
    # 收集所有的分析内容
    all_content = summary + "\n\n"
    for msg in messages:
        if hasattr(msg, 'content'):
            all_content += msg.content + "\n\n"
    
    # 定义数据集信息结构
    dataset_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "analysis_source": "codebase_analysis",
        "codebase_structure": extract_codebase_structure(all_content),
        "data_structure": {
            "raw_data_format": extract_raw_data_format_detailed(all_content),
            "preprocessing_pipeline": extract_preprocessing_pipeline_detailed(all_content),
            "processed_data_format": extract_processed_data_format_detailed(all_content),
            "data_transformations": extract_data_transformations_detailed(all_content)
        },
        "dataloader_organization": {
            "indexing_mechanism": extract_indexing_mechanism_detailed(all_content),
            "sample_structure": extract_sample_structure_detailed(all_content),
            "batch_organization": extract_batch_organization_detailed(all_content),
            "data_relationships": extract_data_relationships_detailed(all_content)
        },
        "data_augmentation": {
            "available_techniques": extract_augmentation_techniques_detailed(all_content),
            "implementation_details": extract_augmentation_implementation_detailed(all_content),
            "application_points": extract_augmentation_points_detailed(all_content)
        },
        "key_files_and_code": extract_key_files_and_code(all_content),
        "training_pipeline": extract_training_pipeline_detailed(all_content),
        "custom_dataset_guidelines": extract_custom_dataset_guidelines_detailed(all_content),
        "code_examples": extract_code_examples(all_content)
    }
    
    return dataset_info

def extract_codebase_structure(content):
    """提取代码库结构信息"""
    structure = {
        "identified_directories": [],
        "key_files": [],
        "file_patterns": []
    }
    
    # 提取目录信息
    import re
    dir_patterns = re.findall(r'(?:directory|folder|path).*?[:\s]([a-zA-Z0-9_/.-]+)', content, re.IGNORECASE)
    structure["identified_directories"] = list(set(dir_patterns))
    
    # 提取文件信息
    file_patterns = re.findall(r'([a-zA-Z0-9_/.-]+\.py)', content)
    structure["key_files"] = list(set(file_patterns))
    
    return structure

def extract_raw_data_format_detailed(content):
    """提取原始数据格式信息 - 详细版本"""
    formats = []
    
    # 从内容中提取具体的文件格式和路径信息
    import re
    
    # 查找文件扩展名和格式描述
    file_extensions = re.findall(r'\.([a-zA-Z0-9]+)', content)
    format_descriptions = re.findall(r'(?:format|file type|data type)[:\s]*([^.\n]+)', content, re.IGNORECASE)
    
    # 查找具体的数据文件示例
    data_file_examples = re.findall(r'([a-zA-Z0-9_.-]+\.(txt|csv|json|bin|idx|wav|jpg|png))', content, re.IGNORECASE)
    
    # 查找数据格式相关的代码片段
    code_snippets = re.findall(r'```[a-zA-Z]*\n(.*?)\n```', content, re.DOTALL)
    
    if data_file_examples:
        formats.append({
            "type": "identified_file_formats",
            "examples": [f"{example[0]}" for example in data_file_examples[:10]],
            "extensions": list(set([example[1] for example in data_file_examples])),
            "source": "extracted_from_codebase_analysis"
        })
    
    if format_descriptions:
        formats.append({
            "type": "format_descriptions",
            "descriptions": format_descriptions[:5],
            "source": "analysis_content"
        })
    
    return formats if formats else [{"type": "no_specific_format_identified", "note": "Check the full analysis for format details"}]

def extract_preprocessing_pipeline_detailed(content):
    """提取预处理管道信息 - 详细版本"""
    pipeline = []
    
    import re
    
    # 查找预处理相关的函数和类
    preprocess_functions = re.findall(r'(?:def|class)\s+([a-zA-Z0-9_]*(?:preprocess|transform|load|parse)[a-zA-Z0-9_]*)', content, re.IGNORECASE)
    
    # 查找预处理步骤描述
    process_steps = re.findall(r'(?:step|stage|phase)[:\s]*([^.\n]+)', content, re.IGNORECASE)
    
    # 查找工具和命令
    tools_commands = re.findall(r'(?:command|tool|script)[:\s]*([a-zA-Z0-9_.-]+)', content, re.IGNORECASE)
    
    if preprocess_functions:
        pipeline.append({
            "step": "preprocessing_functions",
            "identified_functions": preprocess_functions[:10],
            "description": "Functions and classes related to data preprocessing",
            "source": "code_analysis"
        })
    
    if tools_commands:
        pipeline.append({
            "step": "preprocessing_tools",
            "tools": tools_commands[:5],
            "description": "External tools and commands for data processing",
            "source": "analysis_content"
        })
    
    if process_steps:
        pipeline.append({
            "step": "process_descriptions",
            "steps": process_steps[:5],
            "description": "Described processing steps",
            "source": "analysis_content"
        })
    
    return pipeline if pipeline else [{"step": "check_full_analysis", "note": "Detailed pipeline information available in full analysis"}]

def extract_processed_data_format_detailed(content):
    """提取处理后数据格式 - 详细版本"""
    formats = []
    if ".bin" in content and ".idx" in content:
        formats.append({
            "type": "binary_indexed",
            "files": [".bin (data)", ".idx (index)"],
            "description": "Binary format with index files for efficient loading"
        })
    if "dict." in content:
        formats.append({
            "type": "vocabulary_files",
            "files": ["dict.source.txt", "dict.target.txt"],
            "description": "Vocabulary mappings for token-to-index conversion"
        })
    return formats if formats else [{"type": "unknown", "description": "Format not identified"}]

def extract_data_transformations_detailed(content):
    """提取数据转换信息 - 详细版本"""
    transformations = []
    if "padding" in content.lower():
        transformations.append("Sequence padding for batch processing")
    if "eos" in content.lower():
        transformations.append("EOS token handling")
    if "bos" in content.lower():
        transformations.append("BOS token handling")
    return transformations if transformations else ["Transformations not clearly identified"]

def extract_indexing_mechanism_detailed(content):
    """提取索引机制信息 - 详细版本"""
    mechanisms = []
    
    import re
    
    # 查找索引相关的类和函数
    index_classes = re.findall(r'(?:class|def)\s+([a-zA-Z0-9_]*(?:Index|Dataset|Loader)[a-zA-Z0-9_]*)', content, re.IGNORECASE)
    
    # 查找__getitem__方法
    getitem_methods = re.findall(r'def __getitem__\(.*?\):(.*?)(?=def|\Z)', content, re.DOTALL)
    
    if index_classes:
        mechanisms.append({
            "type": "identified_classes",
            "classes": index_classes[:10],
            "description": "Classes related to indexing and data access",
            "source": "code_analysis"
        })
    
    if getitem_methods:
        mechanisms.append({
            "type": "getitem_implementations",
            "count": len(getitem_methods),
            "description": "Found __getitem__ method implementations",
            "sample_code": getitem_methods[0][:200] + "..." if getitem_methods else "",
            "source": "code_analysis"
        })
    
    return mechanisms if mechanisms else [{"type": "check_full_analysis", "note": "Indexing details available in full analysis"}]

def extract_sample_structure_detailed(content):
    """提取样本结构信息 - 详细版本"""
    structure = {}
    
    # 检查常见字段模式
    common_fields = {
        "id": ["id", "index", "identifier", "key"],
        "input": ["input", "data", "features", "x", "source"],
        "target": ["target", "label", "y", "output", "ground_truth"],
        "metadata": ["metadata", "info", "attributes", "properties"]
    }
    
    for field_type, patterns in common_fields.items():
        if any(pattern in content.lower() for pattern in patterns):
            structure[field_type] = {
                "description": f"{field_type.title()} field containing relevant data",
                "common_names": patterns,
                "usage": f"Used for {field_type} in the model pipeline"
            }
    
    # 检查批次结构
    batch_patterns = ["batch", "collate", "stack", "pad"]
    if any(pattern in content.lower() for pattern in batch_patterns):
        structure["batch_format"] = {
            "description": "Batched data structure for efficient processing",
            "common_operations": ["padding", "stacking", "collation"],
            "shape": "Typically (batch_size, ...)"
        }
    
    return structure if structure else {"structure": "generic_sample", "note": "Standard sample structure with input/output fields"}

def extract_batch_organization_detailed(content):
    """提取批次组织信息 - 详细版本"""
    batch_info = {}
    if "collate" in content.lower():
        batch_info["collation"] = "Uses collate function to merge samples into batches"
    if "pad" in content.lower():
        batch_info["padding"] = "Sequences padded to same length within batch"
    if "sort" in content.lower():
        batch_info["sorting"] = "Samples sorted by length for efficiency"
    
    return batch_info if batch_info else {"organization": "unknown"}

def extract_data_relationships_detailed(content):
    """提取数据关系信息 - 详细版本"""
    relationships = []
    if "source" in content and "target" in content:
        relationships.append("Source-target sequence pairs aligned line by line")
    if "alignment" in content.lower():
        relationships.append("Word-level alignment between source and target")
    return relationships if relationships else ["Relationships not clearly identified"]

def extract_augmentation_techniques_detailed(content):
    """提取数据增强技术 - 详细版本"""
    techniques = []
    
    # 图像增强
    image_aug_patterns = ["rotate", "flip", "crop", "resize", "color", "brightness", "contrast", "blur"]
    if any(pattern in content.lower() for pattern in image_aug_patterns):
        techniques.append({
            "type": "image_augmentation",
            "methods": ["rotation", "flipping", "cropping", "color adjustment"],
            "purpose": "Increase dataset diversity and model robustness"
        })
    
    # 文本增强
    text_aug_patterns = ["synonym", "paraphrase", "backtranslation", "noise", "dropout", "mask"]
    if any(pattern in content.lower() for pattern in text_aug_patterns):
        techniques.append({
            "type": "text_augmentation", 
            "methods": ["synonym replacement", "paraphrasing", "noise injection"],
            "purpose": "Improve text model generalization"
        })
    
    # 音频增强
    audio_aug_patterns = ["pitch", "speed", "noise", "echo", "reverb", "time_stretch"]
    if any(pattern in content.lower() for pattern in audio_aug_patterns):
        techniques.append({
            "type": "audio_augmentation",
            "methods": ["pitch shifting", "speed change", "noise addition"],
            "purpose": "Enhance audio model robustness"
        })
    
    # 通用增强
    general_aug_patterns = ["augment", "transform", "random", "jitter", "perturbation"]
    if any(pattern in content.lower() for pattern in general_aug_patterns):
        techniques.append({
            "type": "general_augmentation",
            "methods": ["random transformations", "data perturbation"],
            "purpose": "General data augmentation techniques"
        })
    
    return techniques if techniques else [{"type": "no_augmentation", "note": "No specific augmentation techniques identified"}]

def extract_augmentation_implementation_detailed(content):
    """提取数据增强实现细节 - 详细版本"""
    implementation = []
    if "compute_mask_indices" in content:
        implementation.append("Mask indices computation for token masking")
    if "noising" in content.lower():
        implementation.append("Noising transforms for data augmentation")
    return implementation if implementation else ["Implementation details not clearly identified"]

def extract_augmentation_points_detailed(content):
    """提取数据增强应用点 - 详细版本"""
    points = []
    if "dataset" in content.lower() and "transform" in content.lower():
        points.append("Dataset level transformations")
    if "batch" in content.lower() and "transform" in content.lower():
        points.append("Batch level transformations")
    return points if points else ["Application points not clearly identified"]

def extract_key_files_and_code(content):
    """提取关键文件和代码信息 - 增强版"""
    key_info = {
        "dataset_files": [],
        "training_files": [],
        "config_files": [],
        "code_snippets": [],
        "class_definitions": [],
        "function_definitions": [],
        "import_statements": []
    }
    
    import re
    
    # 提取数据集相关文件
    dataset_files = re.findall(r'([a-zA-Z0-9_/.-]*(?:dataset|data|loader)[a-zA-Z0-9_/.-]*\.py)', content, re.IGNORECASE)
    key_info["dataset_files"] = list(set(dataset_files))
    
    # 提取训练相关文件
    training_files = re.findall(r'([a-zA-Z0-9_/.-]*(?:train|trainer|model)[a-zA-Z0-9_/.-]*\.py)', content, re.IGNORECASE)
    key_info["training_files"] = list(set(training_files))
    
    # 提取配置文件
    config_files = re.findall(r'([a-zA-Z0-9_/.-]*(?:config|setting|cfg)[a-zA-Z0-9_/.-]*\.[a-zA-Z]+)', content, re.IGNORECASE)
    key_info["config_files"] = list(set(config_files))
    
    # 提取类定义
    class_definitions = re.findall(r'class\s+([a-zA-Z0-9_]+)(?:\([^)]*\))?:', content)
    key_info["class_definitions"] = list(set(class_definitions))
    
    # 提取函数定义
    function_definitions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', content)
    key_info["function_definitions"] = list(set(function_definitions))
    
    # 提取import语句
    import_statements = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+([\w., ]+)', content)
    key_info["import_statements"] = [imp.strip() for imp in import_statements[:10]]
    
    # 提取代码片段 - 增加更多类型
    code_patterns = [
        (r'```(?:python|py)?\n(.*?)\n```', 'code_block'),
        (r'def __getitem__\(.*?\):(.*?)(?=def|\n\n|\Z)', 'getitem_method'),
        (r'def __init__\(.*?\):(.*?)(?=def|\n\n|\Z)', 'init_method'),
        (r'def collate.*?\(.*?\):(.*?)(?=def|\n\n|\Z)', 'collate_function'),
        (r'class.*?Dataset.*?:(.*?)(?=class|\Z)', 'dataset_class')
    ]
    
    for pattern, snippet_type in code_patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        for i, match in enumerate(matches[:5]):  # 限制每种类型的数量
            key_info["code_snippets"].append({
                "type": snippet_type,
                "id": f"{snippet_type}_{i+1}",
                "code": match.strip() if isinstance(match, str) else match[0].strip()
            })
    
    return key_info

def extract_training_pipeline_detailed(content):
    """提取训练管道信息 - 详细版本"""
    pipeline = {}
    if "fairseq-train" in content:
        pipeline["training_command"] = "fairseq-train"
    if "batch" in content.lower():
        pipeline["batching"] = "Dynamic batching based on sequence length"
    if "loss" in content.lower():
        pipeline["loss_computation"] = "Cross-entropy loss for sequence prediction"
    
    return pipeline if pipeline else {"pipeline": "not clearly identified"}

def extract_custom_dataset_guidelines_detailed(content):
    """提取自定义数据集指南 - 详细版本"""
    guidelines = {}
    if "fairseq-preprocess" in content:
        guidelines["preprocessing"] = "Use fairseq-preprocess to binarize your data"
    if "source-lang" in content and "target-lang" in content:
        guidelines["language_specification"] = "Specify source and target languages"
    if "dict." in content:
        guidelines["vocabulary"] = "Dictionary files will be generated automatically"
    
    return guidelines if guidelines else {"guidelines": "not clearly identified"}

def extract_code_examples(content):
    """提取代码示例"""
    examples = []
    
    import re
    
    # 提取所有代码块
    code_blocks = re.findall(r'```(?:python|py|bash|shell)?\n(.*?)\n```', content, re.DOTALL)
    
    for i, code in enumerate(code_blocks[:50]):  # 限制数量
        examples.append({
            "example_id": f"code_block_{i+1}",
            "code": code.strip(),
            "context": "extracted_from_analysis",
            "relevance": "data_processing_or_training"
        })
    
    return examples

class CodeAnalysisAgent:
    """代码分析代理类，包含专业的RAG历史记录系统"""
    
    def __init__(self, chat_model, reason_model, embedding_model = None, chroma_persist_directory="./chroma_db"):
        """初始化代理，包括模型和RAG系统"""
        # 初始化模型
        self.tool_model = chat_model#ChatDeepSeek(model="deepseek-chat")
        self.reason_model = reason_model#ChatDeepSeek(model="deepseek-reasoner")
        
        # 初始化嵌入模型
        self.embeddings = embedding_model#self._initialize_embeddings()
        
        # 初始化向量数据库
        self.chroma_persist_directory = chroma_persist_directory
        self.vectorstore = self._initialize_vectorstore()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # RAG检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # RAG chain
        self.rag_chain = self._create_rag_chain()
        
        # 历史记录存储
        self.analysis_history = []
        
        # 工具模型代理（将在action中动态创建）
        self.tool_agent = None
        
        print("CodeAnalysisAgent initialized with professional RAG system")
    
    # def _initialize_embeddings(self):
    #     """初始化嵌入模型，优先使用Google多语言嵌入"""
    #     # 首先尝试Google多语言嵌入模型
    #     try:
    #         from langchain_mistralai import MistralAIEmbeddings
    #         os.environ["MISTRAL_API_KEY"] = "ajdw9kGwkGNh6QwSdFa1jAlpP7TEr45t"
    #         embeddings = MistralAIEmbeddings(
    #             model="mistral-embed",
    #         )
    #         return embeddings
    #     except Exception as e:
    #         print(f"MistralAIEmbeddings embeddings failed: {e}, trying fallback...")
    #     try:
    #         print("Using local HuggingFace embeddings...")
    #         from langchain_huggingface import HuggingFaceEmbeddings
    #         return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #     except ImportError:
    #         print("Warning: No embedding model available")
    #         return None
    
    def _initialize_vectorstore(self):
        """初始化向量数据库"""
        if self.embeddings is None:
            print("Warning: No embeddings available, using in-memory storage")
            return None
        
        return Chroma(
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings
        )
    
    def _create_rag_chain(self):
        """创建RAG链"""
        if self.vectorstore is None:
            return None
        
        # 创建提示模板
        template = """You are an expert code analysis assistant. Use the following pieces of context from previous analyses to enhance your current analysis.

Context from previous analyses:
{context}

Current Question: {question}

Based on the historical context and current question, provide an enhanced analysis that builds upon previous insights while addressing the specific current requirements.

Enhanced Analysis:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # 格式化上下文的函数
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 创建RAG链
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.reason_model
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _store_analysis_history(self, target_path, prompt, final_answer, analysis_summary, discovered_paths):
        """存储分析历史记录到向量数据库"""
        # 创建文档
        analysis_content = f"""
        Target Path: {target_path}
        Prompt: {prompt}
        Analysis Summary: {analysis_summary}
        Final Answer: {final_answer}
        Discovered Paths: {', '.join(discovered_paths)}
        Timestamp: {datetime.datetime.now().isoformat()}
        """
        
        # 存储到历史记录
        history_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "target_path": target_path,
            "prompt": prompt,
            "final_answer": final_answer,
            "analysis_summary": analysis_summary,
            "discovered_paths": discovered_paths
        }
        
        self.analysis_history.append(history_record)
        
        # 保持历史记录在合理大小（最多保存20条）
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
        
        # 如果向量数据库可用，存储到向量数据库
        if self.vectorstore is not None:
            try:
                # 分割文档
                docs = self.text_splitter.create_documents([analysis_content])
                
                # 添加元数据
                for doc in docs:
                    doc.metadata = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "target_path": target_path,
                        "type": "analysis_history"
                    }
                
                # 添加到向量数据库
                self.vectorstore.add_documents(docs)
                print(f"Stored analysis history in vector store. Total records: {len(self.analysis_history)}")
            except Exception as e:
                print(f"Failed to store in vector database: {e}")
        else:
            print(f"Stored analysis history in memory. Total records: {len(self.analysis_history)}")
    
    def _create_enhanced_prompt(self, original_prompt, target_path):
        """使用RAG系统创建增强的prompt"""
        if self.rag_chain is None:
            # 如果RAG不可用，回退到简单的历史记录增强
            if self.analysis_history:
                history_context = "\n\n--- PREVIOUS ANALYSIS CONTEXT ---\n"
                for record in self.analysis_history[-2:]:  # 使用最近的2条记录
                    history_context += f"Previous Analysis (Path: {record['target_path']}):\n"
                    history_context += f"Summary: {record['analysis_summary'][:200]}...\n\n"
                history_context += "--- END CONTEXT ---\n\n"
                return history_context + original_prompt
            return original_prompt
        
        try:
            # 使用RAG链获取增强的prompt
            enhanced_analysis = self.rag_chain.invoke(original_prompt)
            
            # 组合原始prompt和增强的分析
            enhanced_prompt = f"""
            Current Analysis Request:
            {original_prompt}
            
            Previous Analysis Context:
            {enhanced_analysis}
            Please provide a comprehensive analysis that builds upon the previous insights while addressing the current specific requirements.
            """
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"RAG enhancement failed: {e}, using original prompt")
            return original_prompt
    
    def action(self, target_path="./", additional_prompt=""):
        """执行代码分析，逻辑与原main函数相同，但支持额外prompt和历史记录"""
        
        # 处理路径格式
        if not target_path.endswith('/'):
            target_path += '/'
        
        print(f"Starting analysis for path: {target_path}")
        print(f"Additional prompt length: {len(additional_prompt)} characters")
        
        # 检查目标路径是否存在
        if os.path.exists(target_path):
            print(f"Target directory exists: {os.listdir(target_path)[:10]}")
        else:
            print(f"Warning: Target path {target_path} does not exist, falling back to current directory")
            target_path = "./"
        
        # 使用全局变量以保持与现有代码兼容
        global tool_model, reason_model, tool_agent
        tool_model = self.tool_model
        reason_model = self.reason_model
        
        # 定义工具 - 移除root_dir参数，让模型使用完整路径
        tools = [
            ListDirectoryTool(),
            FileSearchTool(),
            ReadFileTool(),
            WriteFileTool(),
        ]
        
        # 创建内存保存器
        memory = MemorySaver()
        
        # 创建系统提示
        tool_system_message = SystemMessage(content=f"""
        You are a code analysis expert. Your goal is to find and analyze dataset/dataloader code efficiently.
        
        PROJECT CONTEXT:
        - Base Path: {target_path}
        - Task: Analyze data processing pipeline, dataloader implementation, and data format requirements
        
        CRITICAL PATH HANDLING RULES:
        🔥 ALWAYS USE FULL PATHS starting with the base path: {target_path}
        🔥 NEVER use relative paths like "A/data/" - ALWAYS use "{target_path}/A/data/"
        🔥 When in doubt, construct paths as: {target_path} + subdirectory_path
        
        SMART EXPLORATION STRATEGY:
        
        1. **Start with overview**: ListDirectoryTool("{target_path}") to see project structure
        
        2. **Find dataset files quickly**:
           - FileSearchTool("*dataset*.py") in base directory
           - FileSearchTool("*data*.py") in base directory
           - FileSearchTool("*loader*.py") in base directory
           - FileSearchTool("*train*.py") in base directory
        
        3. **Read key files completely**: When you find relevant files, read them fully using EXACT PATHS:
           - FIRST use FileSearchTool to find exact file locations
           - THEN use ReadFileTool with the EXACT path returned by FileSearchTool
           - Example: If FileSearchTool returns "fairseq/fairseq/data/data_utils.py", use ReadFileTool("{target_path}fairseq/fairseq/data/data_utils.py")
           - NEVER guess paths - ALWAYS use FileSearchTool first to get the exact location
        
        4. **Extract code patterns**: Look for:
           - How samples are indexed (def __getitem__)
           - Data transformation pipelines
           - Batch creation (collate_fn)
           - File format handling
        
        5. **Path construction examples**:
           ✅ Correct: ListDirectoryTool("{target_path}fairseq/data/")
           ✅ Correct: FileSearchTool("data_utils.py") → returns "fairseq/fairseq/data/data_utils.py" → use ReadFileTool("{target_path}fairseq/fairseq/data/data_utils.py")
           ❌ Wrong: ListDirectoryTool("fairseq/data/")
           ❌ Wrong: ReadFileTool("fairseq/data/data_utils.py") without first checking FileSearchTool results
           
        🔥 CRITICAL FOR READFILETOOL:
           - NEVER guess file paths
           - ALWAYS use FileSearchTool first to get exact file location
           - Use the EXACT path returned by FileSearchTool (may have nested directories)
           - Add {target_path} prefix to the FileSearchTool result
        
        6. **If path fails, try these alternatives**:
           - {target_path}src/
           - {target_path}data/
           - {target_path}examples/
           - {target_path}lib/
           - Use FileSearchTool to locate files when unsure
        
        ANALYSIS DEPTH:
        - Don't just list files - READ and ANALYZE the code
        - Extract specific function/class names
        - Identify data flow patterns
        - Note preprocessing steps and transformations
        - Find batch organization logic
        
        EFFICIENT WORKFLOW:
        ```
        1. ListDirectoryTool("{target_path}") 
        2. FileSearchTool("*dataset*.py") + FileSearchTool("*data*.py")
        3. For each relevant file found:
           a) Note the EXACT path returned by FileSearchTool
           b) Use ReadFileTool("{target_path}" + exact_path_from_search)
           c) Example: FileSearchTool returns "fairseq/fairseq/data/data_utils.py" 
              → Use ReadFileTool("{target_path}fairseq/fairseq/data/data_utils.py")
        4. Extract specific code patterns and class structures
        ```
        
        FOCUS: Find dataset classes, read their implementation, understand data flow.
        BE THOROUGH: Read complete files, not just snippets.
        ADAPT: This project structure may differ from common patterns - be flexible.
        
        REMEMBER: ALWAYS prefix paths with {target_path} - this is CRITICAL for success!
        """)
        
        # 创建tool工具代理
        tool_agent = create_react_agent(
            tool_model,
            tools,
            state_modifier=tool_system_message,
            checkpointer=memory
        )
        self.tool_agent = tool_agent
        
        # 创建工作流
        graph = create_workflow(memory)
        
        # 配置
        config = {"configurable": {"thread_id": f"collab_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
        
        # 构建基础prompt
        base_prompt = """You are a code analysis assistant. I need to understand how to apply a new dataset to this machine learning model/framework. Please analyze the codebase to understand: 1) The input data structure and format requirements, 2) The training pipeline and data processing workflow, 3) The dataloader implementation and how it organizes samples, 4) What data preprocessing is required, 5) How to adapt the system for custom datasets. Focus on extracting the data format details, training procedures, and provide guidance on using custom data with this framework."""
        
        # 如果有额外prompt，添加到前面
        if additional_prompt.strip():
            enhanced_base_prompt = additional_prompt.strip() + "\n\n" + base_prompt
        else:
            enhanced_base_prompt = base_prompt
        
        # 使用RAG系统增强prompt
        print(f"🧠 [RAG_ENHANCEMENT] 开始使用RAG系统增强prompt...")
        print(f"🧠 [RAG_ENHANCEMENT] 原始prompt长度: {len(enhanced_base_prompt)} 字符")
        print(f"🧠 [RAG_ENHANCEMENT] 历史记录数量: {len(self.analysis_history)}")
        
        final_prompt = self._create_enhanced_prompt(enhanced_base_prompt, target_path)
        
        print(f"🧠 [RAG_ENHANCEMENT] 增强后prompt长度: {len(final_prompt)} 字符")
        print(f"🧠 [RAG_ENHANCEMENT] 长度增加: {len(final_prompt) - len(enhanced_base_prompt)} 字符")
        
        if len(final_prompt) > len(enhanced_base_prompt):
            print("✅ [RAG_ENHANCEMENT] RAG系统成功增强了prompt")
        else:
            print("⚠️ [RAG_ENHANCEMENT] RAG系统未能增强prompt，使用原始prompt")
        
        # 执行分析
        get_stream_writer()({
            "data": AiChatResultVO(text=f"🚀 启动协作代理系统分析路径: {target_path}").model_dump_json(
                exclude_none=True
            )
        })
        print(f"🚀 初始提示词长度: {len(final_prompt)} 字符")
        print(f"🚀 目标路径: {target_path}")
        print(f"🚀 配置: {config}")
        final_state = None
        step_count = 0
        
        try:
            for step in graph.stream(
                {
                    "messages": [HumanMessage(content=final_prompt)],
                    "information_summary": "",
                    "need_more_info": True,
                    "current_working_path": target_path,
                    "discovered_paths": [],
                    "path_context": {},
                    "target_base_path": target_path
                },
                config,
                stream_mode="values",
            ):
                step_count += 1
                get_stream_writer()({
                    "data": AiChatResultVO(text=f"🔄 步骤 {step_count}: 分析进行中... (发现 {len(step.get('discovered_paths', []))} 个路径)").model_dump_json(
                        exclude_none=True
                    )
                })
                print(f"\n{'='*50}")
                print(f"🔄 [STEP {step_count}] Current node: {step.get('node_name', 'Unknown')}")
                print(f"🔄 [STEP {step_count}] Need more info: {step.get('need_more_info', 'Unknown')}")
                print(f"🔄 [STEP {step_count}] Current path: {step.get('current_working_path', 'Unknown')}")
                print(f"🔄 [STEP {step_count}] Discovered paths: {len(step.get('discovered_paths', []))}")
                
                if step.get("messages") and len(step["messages"]) > 0:
                    print(f"🔄 [STEP {step_count}] Messages count: {len(step['messages'])}")
                    
                    # 显示所有消息的详细信息
                    for i, msg in enumerate(step["messages"]):
                        msg_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                        print(f"    Message {i+1} ({msg.type}): {msg_preview}")
                    
                    latest_msg = step["messages"][-1]
                    print(f"🔄 [STEP {step_count}] Latest message type: {latest_msg.type}")
                    print(f"🔄 [STEP {step_count}] Latest message length: {len(latest_msg.content)} 字符")
                
                print(f"{'='*50}")
                final_state = step
            
            get_stream_writer()({
                "data": AiChatResultVO(text=f"🎉 分析完成! 总共执行了 {step_count} 个步骤").model_dump_json(
                    exclude_none=True
                )
            })
            print(f"\n🎉 Execution completed! Total steps: {step_count}")
            
            # 显示最终状态的详细信息
            if final_state:
                print("\n" + "="*60)
                print("📋 [FINAL_STATE] 最终状态详情:")
                print("="*60)
                print(f"📊 最终消息数量: {len(final_state.get('messages', []))}")
                print(f"📊 信息摘要长度: {len(final_state.get('information_summary', ''))}")
                print(f"📊 发现的路径: {final_state.get('discovered_paths', [])}")
                print(f"📊 是否需要更多信息: {final_state.get('need_more_info', 'Unknown')}")
                print(f"📊 当前工作路径: {final_state.get('current_working_path', 'Unknown')}")
                
                # 显示所有最终消息
                final_messages = final_state.get("messages", [])
                if final_messages:
                    print(f"\n--- 最终对话历史 ({len(final_messages)} 条消息) ---")
                    for i, msg in enumerate(final_messages):
                        print(f"\n消息 {i+1} ({msg.type}):")
                        print(f"长度: {len(msg.content)} 字符")
                        print("内容:")
                        print(msg.content)
                        print("-" * 40)
                
                print("="*60)
                print("📋 [FINAL_STATE] 状态详情结束")
                print("="*60 + "\n")
            
            # 处理结果
            if final_state:
                final_messages = final_state.get("messages", [])
                final_summary = final_state.get("information_summary", "")
                discovered_paths = final_state.get("discovered_paths", [])
                
                final_answer = final_messages[-1].content if final_messages else "No final answer generated"
                
                # 存储到历史记录
                self._store_analysis_history(
                    target_path=target_path,
                    prompt=enhanced_base_prompt,
                    final_answer=final_answer,
                    analysis_summary=final_summary,
                    discovered_paths=discovered_paths
                )
                
                # 构建输出数据
                output_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "target_base_path": target_path,
                    "discovered_paths": discovered_paths,
                    "analysis_summary": final_summary,
                    "final_answer": final_answer,
                    "additional_prompt_used": additional_prompt,
                    "conversation_history": [
                        {
                            "role": msg.type,
                            "content": msg.content
                        }
                        for msg in final_messages
                    ]
                }
                
                # 保存结果文件
                project_name = os.path.basename(os.path.abspath(target_path.rstrip('/'))) or "project"
                output_file = f"{project_name}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                get_stream_writer()({
                    "data": AiChatResultVO(text=f"📁 分析结果已保存到文件: {output_file}").model_dump_json(
                        exclude_none=True
                    )
                })
                print(f"Analysis results saved to file: {output_file}")
                
                # 生成数据集信息文件
                dataset_info = extract_dataset_info_from_analysis(final_messages, final_summary)
                dataset_info["target_base_path"] = target_path
                dataset_info["discovered_paths"] = discovered_paths
                dataset_info["additional_prompt_used"] = additional_prompt
                
                dataset_info_file = f"{project_name}_dataset_info_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(dataset_info_file, "w", encoding="utf-8") as f:
                    json.dump(dataset_info, f, ensure_ascii=False, indent=2)
                
                get_stream_writer()({
                    "data": AiChatResultVO(text=f"📊 数据集信息已保存到文件: {dataset_info_file}").model_dump_json(
                        exclude_none=True
                    )
                })
                print(f"Dataset-specific information saved to file: {dataset_info_file}")
                
                # 生成Markdown报告
                markdown_content = f"""# {project_name.title()} Analysis Report
Generation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Path: {target_path}
Additional Prompt: {additional_prompt[:1000] + '...' if len(additional_prompt) > 1000 else additional_prompt}
Discovered Paths: {', '.join(discovered_paths[:5])}
Historical Context Used: {'Yes' if len(self.analysis_history) > 1 else 'No'}

## Analysis Summary
{final_summary}

## Final Conclusion
{final_answer}
"""
                
                markdown_file = f"{project_name}_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(markdown_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                
                get_stream_writer()({
                    "data": AiChatResultVO(text=f"📝 简洁报告已保存到文件: {markdown_file}").model_dump_json(
                        exclude_none=True
                    )
                })
                print(f"Concise report saved to file: {markdown_file}")
                
                # 读取生成的文件内容
                file_contents = {}
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        file_contents["analysis_json"] = json.load(f)
                except Exception as e:
                    print(f"Failed to read {output_file}: {e}")
                    file_contents["analysis_json"] = None
                
                try:
                    with open(dataset_info_file, "r", encoding="utf-8") as f:
                        file_contents["dataset_info_json"] = json.load(f)
                except Exception as e:
                    print(f"Failed to read {dataset_info_file}: {e}")
                    file_contents["dataset_info_json"] = None
                
                try:
                    with open(markdown_file, "r", encoding="utf-8") as f:
                        file_contents["markdown_report"] = f.read()
                except Exception as e:
                    print(f"Failed to read {markdown_file}: {e}")
                    file_contents["markdown_report"] = None
                
                return {
                    "success": True,
                    "final_answer": final_answer,
                    "analysis_summary": final_summary,
                    "discovered_paths": discovered_paths,
                    "output_files": [output_file, dataset_info_file, markdown_file],
                    "file_contents": file_contents,
                    "history_records_count": len(self.analysis_history),
                    "markdown":markdown_content,
                    "json_out":dataset_info,
                    "summary":output_data
                }
            else:
                return {
                    "success": False,
                    "error": "No final state generated",
                    "history_records_count": len(self.analysis_history)
                }
                
        except Exception as e:
            get_stream_writer()({
                "data": AiChatResultVO(text=f"❌ 分析过程中发生错误: {str(e)}").model_dump_json(
                    exclude_none=True
                )
            })
            print(f"Error during analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "history_records_count": len(self.analysis_history)
            }
    
    def get_history_summary(self):
        """获取历史记录摘要"""
        if not self.analysis_history:
            return "No historical analysis records available."
        
        summary = f"Historical Analysis Records: {len(self.analysis_history)} total\n\n"
        
        for i, record in enumerate(self.analysis_history[-20:], 1):  # 显示最近5条
            summary += f"{i}. {record['timestamp'][:19]}\n"
            summary += f"   Path: {record['target_path']}\n"
            summary += f"   Summary: {record['analysis_summary'][:100]}...\n\n"
        
        return summary
    
    def clear_history(self):
        """清除历史记录"""
        self.analysis_history.clear()
        
        # 清除向量数据库
        if self.vectorstore is not None:
            try:
                # 重新初始化向量数据库
                import shutil
                if os.path.exists(self.chroma_persist_directory):
                    shutil.rmtree(self.chroma_persist_directory)
                self.vectorstore = self._initialize_vectorstore()
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 3}
                ) if self.vectorstore else None
                self.rag_chain = self._create_rag_chain()
                print("Analysis history and vector database cleared.")
            except Exception as e:
                print(f"Failed to clear vector database: {e}")
        else:
            print("Analysis history cleared.")
    
    def search_history(self, query, top_k=3):
        """搜索历史记录"""
        if self.vectorstore is None:
            # 简单的文本搜索回退
            results = []
            query_lower = query.lower()
            for record in self.analysis_history:
                content = f"{record['prompt']} {record['analysis_summary']} {record['final_answer']}".lower()
                if query_lower in content:
                    results.append(record)
            return results[:top_k]
        
        try:
            # 使用向量搜索
            docs = self.vectorstore.similarity_search(query, k=top_k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []
    
    def get_rag_stats(self):
        """获取RAG系统统计信息"""
        stats = {
            "memory_records": len(self.analysis_history),
            "vector_store_available": self.vectorstore is not None,
            "embeddings_available": self.embeddings is not None,
            "rag_chain_available": self.rag_chain is not None,
            "persist_directory": self.chroma_persist_directory
        }
        
        if self.vectorstore is not None:
            try:
                # 尝试获取向量数据库统计信息
                collection = self.vectorstore._collection
                stats["vector_store_count"] = collection.count()
            except Exception as e:
                stats["vector_store_count"] = "unknown"
                stats["vector_store_error"] = str(e)
        
        return stats





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

# 📦 datapter

## 🔍 项目简介

`datapter` 是一个基于多 Agent 协作机制的智能代码适配工具，旨在解决不同模型（模型 A → 模型 B）之间 **数据集结构不兼容** 的问题。项目通过调用大模型（如 LLM）理解源数据格式并自动生成目标模型可用的适配代码，使 AI 工程师免于繁琐的手动编写转换脚本。

该系统广泛适用于 AI 模型迁移、开源模型重用、数据格式转换等场景。

---

## 🧠 核心功能

* 🧩 数据结构理解与抽象（支持 LLM 自动推理）
* 🔁 源格式 → 中间表示 → 目标格式 的自动映射转换
* 🧱 模块化适配器生成（支持分阶段调用）
* ✅ 自动测试与验证机制（规划中）

---

## 🏗️ 项目结构

项目位于 `src/` 目录下，遵循清晰的 DDD（领域驱动设计）和多智能体模式：

```
datapter/
├── adapter/                  # 对接层（接口适配、日志）
│   └── vo/                   # 输入输出数据对象
├── app/                      # 服务层（Agent 编排）
├── domain/                  
│   ├── constant/             # 常量定义
│   └── model/                # 各类智能体模型
│       ├── coordinator_agent.py  # 调度与任务分解
│       ├── demo_agent.py         # 示例适配 Agent
│       └── demo2_agent.py        # 备用适配逻辑
├── utils/                    # 工具类（大模型调用、ID 生成等）
├── main.py                   # 项目入口
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 推荐使用虚拟环境
pip install -r requirements.txt
```

或使用 `conda`：

```bash
conda env create -f environment.yml
conda activate datapter
```

### 2. 运行示例

```bash
python main.py
```

### 3. 测试curl

```curl
curl --request POST \
  --url http://localhost:8080/aichat/chat \
  --header 'Accept: */*' \
  --header 'Accept-Encoding: gzip, deflate, br' \
  --header 'Connection: keep-alive' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: PostmanRuntime-ApipostRuntime/1.1.0' \
  --data '{
    "conversation_id": "1919358918656524289",
    "messages": [
        {
            "role": "user",
            "content": "code and model limit 10 world"
        }
    ]
}'
```

可在配置中自定义目标模型、适配策略、日志等级等参数。

---

## 🧬 Agent 模块说明

| Agent 文件               | 功能描述                 |
| ---------------------- | -------------------- |
| `coordinator_agent.py` | 协调多个子 Agent，完成适配任务调度 |
| `demo_agent.py`        | 简单规则映射的演示适配器         |
| `demo2_agent.py`       | 更复杂或多阶段适配逻辑的实现       |

---

## 📦 示例能力

* ✅ JSON → CSV → PyTorch Dataset
* ✅ COCO → YOLO → MMDetection
* ✅ 结构化文本 → Prompt 模板 → Transformers 格式
* ✅ 自定义字段提取与重组

---

## 🛠️ 技术栈

* Python 3.12+
* FastAPI（如用于后续服务化）
* OpenAI / LLM 接口（如 `llm_util.py` 中封装）
* Snowflake ID 生成器（见 `SnowFlake.py`）

---

## 🔮 未来规划

* Web UI 界面可视化转换过程
* 支持更多数据结构类型（表格、图结构、图像标注等）
* 任务链路图 + 数据血缘跟踪能力
* 可插拔式插件系统（支持社区模型格式）

---

## 📄 License

本项目采用 MIT 协议开源。
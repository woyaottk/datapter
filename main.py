import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Set

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from src.utils.router_manager import ProtocolType, RouterManager

# 加载 .env 文件
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
TEST_VAR = os.getenv("TEST_VAR")
print(f"TEST_VAR: {TEST_VAR}")
# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


class Server:
    """应用程序启动器类"""

    def __init__(self):
        """初始化启动器"""
        print("FastAPI 应用程序已初始化")
        self._signal_handlers: Set[signal.Signals] = set()
        self.router_manager = RouterManager(gateway_port=8080)
        
        # 添加CORS中间件
        self.router_manager.gateway_app.add_middleware(
            CORSMiddleware,
            allow_origins=[os.getenv("FRONTEND_URL")],  # 允许的前端源
            allow_credentials=True,
            allow_methods=["*"],  # 允许所有HTTP方法
            allow_headers=["*"],  # 允许所有请求头
        )

    async def register_router(self):
        from src.adapter.ai_chat_router import router as ai_chat

        self.router_manager.register_router(
            router=ai_chat, name="ai_chat", protocol=[ProtocolType.GATEWAY]
        )

    async def run(self):
        """异步运行应用程序"""
        try:
            await self.register_router()

            try:
                await self.router_manager.start()
            finally:
                await self.router_manager.shutdown()
        except Exception as e:
            print(f"应用程序运行失败: {str(e)}")
            raise


def main():
    """主函数"""
    try:
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        starter = Server()
        loop.run_until_complete(starter.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()

import asyncio
from enum import Enum
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, APIRouter


class ProtocolType(str, Enum):
    """Protocol types supported by the service"""

    RPC = "PRC"
    GATEWAY = "gateway"


class RouterHelper:
    def __init__(self):
        self._routers: Dict[str, APIRouter] = {}
        self._router_prefixes: Dict[str, str] = {}

    def register_router(self, router: APIRouter, name: str, prefix: str = None):
        """Register a router with optional prefix"""
        if name in self._routers:
            raise ValueError(f"Router with name '{name}' already registered")

        self._routers[name] = router
        if prefix:
            self._router_prefixes[name] = prefix

    def get_router(self, name: str) -> APIRouter:
        """Get a registered router by name"""
        if name not in self._routers:
            raise ValueError(f"Router '{name}' not found")
        return self._routers[name]

    def get_all_routers(self) -> List[APIRouter]:
        """Get all registered routers"""
        return list(self._routers.values())

    def get_router_with_prefix(self, name: str) -> tuple[APIRouter, str]:
        """Get router and its prefix by name"""
        router = self.get_router(name)
        prefix = self._router_prefixes.get(name)
        return router, prefix


class RouterManager:
    def __init__(
        self,
        rpc_host: str = "0.0.0.0",
        rpc_port: int = 3001,
        gateway_host: str = "0.0.0.0",
        gateway_port: int = 3000,
        rpc_title: str = "RPC API Service",
        gateway_title: str = "Gateway Service",
    ):
        """
        Initialize the ServiceManager with configurable protocols and ports.

        Args:
            rpc_host: Host address for the rpc service
            rpc_port: Port number for the rpc service
            gateway_host: Host address for the gateway service
            gateway_port: Port number for the gateway service
            rpc_title: Title for the rpc FastAPI app
            gateway_title: Title for the gateway FastAPI app
        """
        self.rpc_host = rpc_host
        self.rpc_port = rpc_port
        self.rpc_router_manager = RouterHelper()
        self.rpc_app = FastAPI(title=rpc_title)
        self.rpc_server = None

        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.gateway_router_manager = RouterHelper()
        self.gateway_app = FastAPI(title=gateway_title)
        self.gateway_server = None

        # Store protocol states
        self._protocols: Dict[ProtocolType, bool] = {
            ProtocolType.RPC: False,
            ProtocolType.GATEWAY: False,
        }

        # Control flags for graceful shutdown
        self._running = False
        self._stop_event = asyncio.Event()

    def register_router(
        self,
        router: APIRouter,
        name: str,
        protocol: ProtocolType | list[ProtocolType] = None,
        prefix: str = None,
    ):
        """
        Register a service with specified protocols

        Args:
            router: The APIRouter containing the service endpoints
            name: Unique name for the service
            protocol: Protocol(s) to expose the service on (REST and/or RPC)
            prefix: Optional URL prefix for the service
        """
        if protocol is None:
            protocol = [ProtocolType.RPC]
        protocols = [protocol] if isinstance(protocol, ProtocolType) else protocol

        for proto in protocols:
            if proto == ProtocolType.RPC:
                self._register_rpc_router(router, name, prefix)
                self._protocols[ProtocolType.RPC] = True
            elif proto == ProtocolType.GATEWAY:
                self._register_gateway_router(router, name, prefix)
                self._protocols[ProtocolType.GATEWAY] = True

    def _register_rpc_router(self, router: APIRouter, name: str, prefix: str = None):
        """Register a service for inner rpc protocol"""
        self.rpc_router_manager.register_router(router, name, prefix)
        if prefix:
            self.rpc_app.include_router(router, prefix=prefix)
        else:
            self.rpc_app.include_router(router)

    def _register_gateway_router(
        self, router: APIRouter, name: str, prefix: str = None
    ):
        """Register a service for gateway protocol"""
        self.gateway_router_manager.register_router(router, name, prefix)
        if prefix:
            self.gateway_app.include_router(router, prefix=prefix)
        else:
            self.gateway_app.include_router(router)

    def get_router(self, name: str, protocol: ProtocolType) -> APIRouter:
        """
        Get a registered service by name and protocol

        Args:
            name: Name of the service to retrieve
            protocol: Protocol of the service (GATEWAY or RPC)
        """
        if protocol == ProtocolType.RPC:
            return self.rpc_router_manager.get_router(name)
        elif protocol == ProtocolType.GATEWAY:
            return self.gateway_router_manager.get_router(name)
        raise ValueError(f"Unsupported protocol: {protocol}")

    async def _run_rpc(self):
        """Internal method to run the inner RPC service"""
        config = uvicorn.Config(self.rpc_app, host=self.rpc_host, port=self.rpc_port)
        self.rpc_server = uvicorn.Server(config)

        # Create tasks for both server and stop event
        server_task = asyncio.create_task(self.rpc_server.serve())
        stop_task = asyncio.create_task(self._stop_event.wait())

        # Wait for either server completion or stop event
        done, pending = await asyncio.wait(
            [server_task, stop_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # If stop event triggered, initiate shutdown
        if server_task not in done:
            await self.rpc_server.shutdown()

    async def _run_gateway(self):
        """Internal method to run the GATEWAY service"""
        config = uvicorn.Config(
            self.gateway_app, host=self.gateway_host, port=self.gateway_port
        )
        self.gateway_server = uvicorn.Server(config)

        # Create tasks for both server and stop event
        server_task = asyncio.create_task(self.gateway_server.serve())
        stop_task = asyncio.create_task(self._stop_event.wait())

        # Wait for either server completion or stop event
        done, pending = await asyncio.wait(
            [server_task, stop_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # If stop event triggered, initiate shutdown
        if server_task not in done:
            await self.gateway_server.shutdown()

    async def start(self, protocols: list[ProtocolType] = None):
        """
        Start services with specified protocols in separate tasks.

        Args:
            protocols: List of protocols to start. If None, starts all registered protocols.
        """
        if protocols is None:
            protocols = [proto for proto, enabled in self._protocols.items() if enabled]

        self._running = True
        tasks = []

        for protocol in protocols:
            if protocol == ProtocolType.RPC and self._protocols[ProtocolType.RPC]:
                tasks.append(asyncio.create_task(self._run_rpc()))
                print(f"RPC router started at http://{self.rpc_host}:{self.rpc_port}")

            elif (
                protocol == ProtocolType.GATEWAY
                and self._protocols[ProtocolType.GATEWAY]
            ):
                tasks.append(asyncio.create_task(self._run_gateway()))
                print(
                    f"Gateway router started at http://{self.gateway_host}:{self.gateway_port}"
                )

        await asyncio.gather(*tasks)

    async def shutdown(self):
        """Gracefully shutdown all services"""
        if not self._running:
            return

        self._running = False

        # Signal all services to stop
        self._stop_event.set()

        # Create shutdown tasks
        shutdown_tasks = []

        if self.gateway_server:
            shutdown_tasks.append(self.gateway_server.shutdown())
        if self.rpc_server:
            shutdown_tasks.append(self.rpc_server.shutdown())

        # Wait for all servers to shutdown
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks)

        # Clean up FastAPI applications
        await self._cleanup_fastapi_app(self.gateway_app)
        await self._cleanup_fastapi_app(self.rpc_app)

        # Reset state
        self.gateway_server = None
        self.rpc_server = None
        self._stop_event.clear()

        print("All services stopped successfully")

    async def _cleanup_fastapi_app(self, app: FastAPI):
        """Clean up FastAPI application resources"""
        if not app:
            return

        # Close any remaining connections
        if hasattr(app, "router"):
            for route in app.router.routes:
                if hasattr(route, "close"):
                    await route.close()

        # Clean up middleware
        if hasattr(app, "middleware_stack"):
            for middleware in app.middleware_stack.middlewares:
                if hasattr(middleware, "close"):
                    await middleware.close()

        # Force cleanup of event handlers
        app.router.lifespan.startup = []
        app.router.lifespan.shutdown = []

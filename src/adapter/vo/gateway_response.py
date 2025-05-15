from typing import TypeVar, Generic, Optional

from pydantic import BaseModel, Field

T = TypeVar("T")


class GatewayResponse(BaseModel, Generic[T]):
    data: Optional[T] = Field(None, description="请求结果")
    code: str = Field("0", description="状态码")
    message: str = Field("调用成功!", description="处理状态消息")
    debug: str = Field("", description="调试信息")

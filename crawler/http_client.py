"""爬虫专用 HTTP 请求工具。

真实网站请求经常会遇到偶发超时、代理断连、远端主动断开等问题。
这里做一个很薄的重试封装，让业务代码专心处理“怎么解析数据”。
"""

from __future__ import annotations

import time
from typing import Any

import requests

from crawler.config import REQUEST_DELAY_SECONDS, REQUEST_HEADERS, REQUEST_TIMEOUT


def get_response(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    delay_seconds: float = REQUEST_DELAY_SECONDS,
    **kwargs: Any,
) -> requests.Response:
    """发送 GET 请求，失败时进行少量重试。

    retries 不宜太大；课程项目的数据量不大，失败时多等几秒比高频请求更稳。
    """

    headers = kwargs.pop("headers", REQUEST_HEADERS)
    timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
    request_kwargs = kwargs
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                **request_kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay_seconds * attempt)

    raise RuntimeError(f"请求失败，已重试 {retries} 次: {url}") from last_error

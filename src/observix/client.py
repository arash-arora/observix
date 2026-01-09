from types import TracebackType
from typing import Optional, Type

import httpx


class Client:
    def __init__(
        self,
        base_url: str = "https://api.example.com",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=base_url,
            headers=self._get_headers(),
            timeout=timeout,
        )

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "obs-sdk-python",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

from dataclasses import field
from typing import Dict, List

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse as SglangChatCompletionResponse,
)


class ChatCompletionResponse(SglangChatCompletionResponse):
    usage_details: List[Dict[str, Dict[str, int]]] = field(default_factory=list)

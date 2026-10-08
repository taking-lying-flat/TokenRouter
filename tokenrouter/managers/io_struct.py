from dataclasses import dataclass
from typing import Dict, List

from sglang.srt.managers.io_struct import BatchStrOut as SglangBatchStrOut
from sglang.srt.managers.io_struct import BatchTokenIDOut as SglangBatchTokenIDOut


@dataclass
class BatchTokenIDOut(SglangBatchTokenIDOut):
    model_tokens: Dict[str, List[int]]


@dataclass
class BatchStrOut(SglangBatchStrOut):
    model_tokens: Dict[str, List[int]]
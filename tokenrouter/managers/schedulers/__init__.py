from typing import Dict, Type

from sglang.srt.managers.scheduler import Scheduler

from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.managers.schedulers.citer_slm_scheduler import CITERSLMScheduler
from tokenrouter.managers.schedulers.collm_base_scheduler import CoLLMBaseScheduler
from tokenrouter.managers.schedulers.glimprouter_llm_scheduler import (
    GlimpRouterLLMScheduler,
)
from tokenrouter.managers.schedulers.glimprouter_slm_scheduler import (
    GlimpRouterSLMScheduler,
)
from tokenrouter.managers.schedulers.per_request_random_routing_scheduler import (
    PerRequestRandomRoutingScheduler,
)
from tokenrouter.managers.schedulers.query_level_routing_entry_scheduler import (
    QueryLevelRoutingEntryScheduler,
)
from tokenrouter.managers.schedulers.r2r_llm_scheduler import R2RLLMScheduler
from tokenrouter.managers.schedulers.r2r_slm_scheduler import R2RSLMScheduler
from tokenrouter.managers.schedulers.random_routing_scheduler import (
    RandomRoutingScheduler,
)
from tokenrouter.managers.schedulers.rstitch_llm_scheduler import RStitchLLMScheduler
from tokenrouter.managers.schedulers.rstitch_slm_scheduler import RStitchSLMScheduler

CITERLLMScheduler = R2RLLMScheduler
CoLLMAssistScheduler = R2RLLMScheduler

SCHEDULER_REGISTRY: Dict[str, Type[Scheduler]] = {
    "sglang": Scheduler,
    "base": BaseTokenRoutingScheduler,
    "r2r_llm": R2RLLMScheduler,
    "r2r_slm": R2RSLMScheduler,
    "citer_slm": CITERSLMScheduler,
    "citer_llm": CITERLLMScheduler,
    "glimprouter_slm": GlimpRouterSLMScheduler,
    "glimprouter_llm": GlimpRouterLLMScheduler,
    "rstitch_llm": RStitchLLMScheduler,
    "rstitch_slm": RStitchSLMScheduler,
    "random_routing": RandomRoutingScheduler,
    "query_level_routing": QueryLevelRoutingEntryScheduler,
    "per_request_random_routing": PerRequestRandomRoutingScheduler,
    "collm_base": CoLLMBaseScheduler,
    "collm_assist": CoLLMAssistScheduler,
}

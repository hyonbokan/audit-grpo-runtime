"""Load explicitly selected gateway credentials before constructing either judge."""

from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from llm_router.config import RouterConfig


def judge_router_config(settings: dict[str, Any], *, semaphore_limit: int) -> RouterConfig:
    env_file = settings.get("judge_env_file")
    if env_file:
        path = Path(env_file)
        names = ("LITELLM_GATEWAY_URL", "LITELLM_GATEWAY_API_KEY")
        if not path.is_file() or not all(dotenv_values(path).get(name) for name in names):
            raise ValueError("Complete gateway credential file required")
        load_dotenv(path, override=True)
    config = RouterConfig(
        enable_langfuse=False, ci_mode=False, openai_semaphore_limit=semaphore_limit
    )
    if env_file and not config.gateway_enabled:
        raise ValueError("Complete gateway credential file required")
    return config

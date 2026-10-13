"""Fresh dotenv gateway credentials override stale container credentials."""

import pytest


@pytest.mark.asyncio
async def test_dotenv_gateway_replaces_stale_credentials_in_actual_sdk_client(
    tmp_path, monkeypatch
):
    from llm_router.clients import ClientPool

    from grpo.judge_environment import judge_router_config

    monkeypatch.setenv("OPENAI_API_KEY", "stale-direct-key")
    monkeypatch.setenv("LITELLM_GATEWAY_URL", "https://old-gateway.invalid")
    monkeypatch.setenv("LITELLM_GATEWAY_API_KEY", "stale-gateway-key")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LITELLM_GATEWAY_URL=https://new-gateway.invalid\nLITELLM_GATEWAY_API_KEY=fresh-test-key\n"
    )
    config = judge_router_config({"judge_env_file": str(env_file)}, semaphore_limit=1)
    pool = ClientPool(config)
    try:
        client = await pool.get_openai_client()
        assert str(client.base_url) == "https://new-gateway.invalid/openai_passthrough/v1/"
        assert client.api_key == "fresh-test-key"
        assert client.max_retries == 0
    finally:
        await pool.close()


@pytest.mark.parametrize("contents", [None, "", "LITELLM_GATEWAY_URL=https://gateway.invalid\n"])
def test_missing_dotenv_gateway_pair_rejects_direct_or_stale_fallback(
    tmp_path, monkeypatch, contents
):
    from grpo.judge_environment import judge_router_config

    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    monkeypatch.setenv("LITELLM_GATEWAY_URL", "https://old-gateway.invalid")
    monkeypatch.setenv("LITELLM_GATEWAY_API_KEY", "old-gateway-key")
    env_file = tmp_path / ".env"
    if contents is not None:
        env_file.write_text(contents)
    with pytest.raises(ValueError, match="gateway credential file"):
        judge_router_config({"judge_env_file": str(env_file)}, semaphore_limit=1)

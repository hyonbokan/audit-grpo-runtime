import compute_estimate as estimate
import pytest

MODELS = estimate.load_models()
GPUS = estimate.load_gpus()


def test_calibration_reproduces_the_measured_a6000_fits():
    qwen, a6000 = MODELS["Qwen3.8-27B"], GPUS["A6000"]
    # Measured 2026-10-22: 36k untiled, no offload 44.7 GiB; 96k tiled, all layers streamed 42.7.
    resident = estimate.update_gib(qwen, a6000, gpus=2, tokens=36_000, tiled=False, offload=False)
    streamed = estimate.update_gib(qwen, a6000, gpus=2, tokens=96_000, tiled=True, offload=True)
    assert resident == pytest.approx(44.7, abs=1.5)
    assert streamed == pytest.approx(42.7, abs=1.5)
    assert estimate.update_gib(qwen, a6000, gpus=2, tokens=128_000, tiled=True, offload=True) > 47.4


def test_bf16_weights_scale_with_every_expert():
    flash = MODELS["GLM-5.3-Flash"]
    assert estimate.weights_gib(flash) == pytest.approx(320e9 * 2 / 2**30)


def test_glm_flash_bf16_rollout_needs_more_than_one_h100_node():
    flash = MODELS["GLM-5.3-Flash"]
    assert not estimate.rollout_fits(flash, GPUS["H100"], gpus=8, sequences=4, tokens=96_000)
    assert estimate.rollout_fits(flash, GPUS["H200"], gpus=8, sequences=4, tokens=96_000)


def test_larger_tiers_never_need_fewer_gpus():
    for model in MODELS.values():
        for gpu in ("H100", "H200", "B200"):
            sizes = [estimate.tier(model, gpu, spec)["gpus"] for spec in estimate.TIERS]
            known = [size for size in sizes if size is not None]
            assert known == sorted(known)


def test_unknown_model_or_gpu_names_the_allowed_values():
    with pytest.raises(KeyError, match=r"GLM-5\.3-Flash"):
        estimate.pick(MODELS, "GLM-9")

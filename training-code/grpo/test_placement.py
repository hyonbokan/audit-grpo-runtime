"""Update placement: one GPU or a layer split, and the GPUs each container may see."""

import pytest

from grpo import placement

ONE = {"gpus": [0], "embedding": "offloaded"}
TWO = {"gpus": [0, 1], "embedding": "gpu", "max_memory_gib": 44}


def test_one_gpu_pins_every_layer_to_the_only_visible_device():
    assert placement.load_options(ONE) == {"device_map": {"": 0}, "offload_embedding": True}
    assert placement.load_options({**ONE, "embedding": "gpu"}) == {
        "device_map": {"": 0},
        "offload_embedding": False,
    }


def test_two_gpus_split_layers_with_unsloths_planner_under_a_per_device_cap():
    assert placement.load_options(TWO) == {
        "device_map": "unsloth",
        "max_memory": {0: "44GiB", 1: "44GiB"},
        "offload_embedding": False,
    }


@pytest.mark.parametrize(
    "invalid,message",
    [
        ({**TWO, "embedding": "offloaded"}, "embedding offload"),
        ({"gpus": [0, 1], "embedding": "gpu"}, "max_memory_gib"),
        ({**TWO, "max_memory_gib": 0}, "max_memory_gib"),
        ({**TWO, "max_memory_gib": True}, "max_memory_gib"),
        ({**ONE, "max_memory_gib": 44}, "max_memory_gib"),
        ({**ONE, "embedding": "cpu"}, "embedding"),
        ({"gpus": [], "embedding": "gpu"}, "gpus"),
        ({"gpus": [0, 0], "embedding": "gpu", "max_memory_gib": 44}, "gpus"),
        ({"gpus": [-1], "embedding": "gpu"}, "gpus"),
        ({"gpus": ["0"], "embedding": "gpu"}, "gpus"),
        ({**ONE, "device_map": "auto"}, "Unknown update_placement keys"),
    ],
)
def test_invalid_placements_fail_with_the_field_to_change(invalid, message):
    with pytest.raises(ValueError, match=message):
        placement.load_options(invalid)


def test_container_sees_exactly_the_listed_gpus():
    assert placement.docker_gpus([0]) == '"device=0"'
    assert placement.docker_gpus([0, 1]) == '"device=0,1"'


def test_rollout_sees_one_gpu_per_tensor_parallel_rank():
    assert placement.rollout_gpus({"dtype": "bfloat16"}) == [0]
    assert placement.rollout_gpus({"tensor_parallel_size": 2}) == [0, 1]
    for invalid in (0, 3, True, 1.0):
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            placement.rollout_gpus({"tensor_parallel_size": invalid})


class Parameter:
    def __init__(self, device):
        self.device = device


class Layer:
    def __init__(self, *devices):
        self.devices = devices

    def parameters(self):
        return iter(Parameter(device) for device in self.devices)


def test_layer_devices_counts_each_layer_once_by_where_its_weights_live():
    layers = [Layer("cuda:0", "cuda:0"), Layer("cuda:1"), Layer("cuda:1"), Layer("cpu")]
    assert placement.layer_devices(layers) == {"cpu": 1, "cuda:0": 1, "cuda:1": 2}


def test_a_layer_whose_weights_span_devices_is_rejected():
    with pytest.raises(ValueError, match="layer 1"):
        placement.layer_devices([Layer("cuda:0"), Layer("cuda:0", "cuda:1")])


def test_final_hidden_states_move_to_the_head_device_and_the_move_is_reported():
    torch = pytest.importorskip("torch")
    hidden = torch.ones(3, 4, requires_grad=True)
    moved, transfer = placement.hidden_on_head_device(hidden, torch.empty(5, 4, device="meta"))
    assert moved.device.type == "meta" and moved.shape == (3, 4)
    assert transfer == "cpu->meta"


def test_final_hidden_states_already_on_the_head_device_are_returned_unchanged():
    torch = pytest.importorskip("torch")
    hidden = torch.ones(3, 4, requires_grad=True)
    same, transfer = placement.hidden_on_head_device(hidden, torch.zeros(5, 4))
    assert same is hidden and transfer is None

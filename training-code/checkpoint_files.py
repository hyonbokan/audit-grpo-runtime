"""The files of a saved checkpoint: which a resume needs and which only a resume reads."""

ADAPTER = "adapter_model.safetensors"
# Read only to continue training; retention and thin backups may drop them.
RESUME_ONLY = ("optimizer.pt", "scheduler.pt", "rng_state.pth")
# Everything an update or a restore loads to continue from a checkpoint.
RESUMABLE = (ADAPTER, "adapter_config.json", *RESUME_ONLY, "trainer_state.json")

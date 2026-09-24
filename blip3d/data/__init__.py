"""Training data: manifests -> per-modality tasks -> a rank-synced task mixture -> batches (docs/INTERFACES.md §1).

    from blip3d.data import build_mixture
    mix = build_mixture("recipes/data/v12_ss.yaml", per_gpu_bs=4, seed=0, resume_step=step,
                        align_points=True, qwen_path=qwen_dir)
    loader = DataLoader(mix.dataset, batch_size=4, collate_fn=mix.collate_fn, num_workers=mix.num_workers)

TRELLIS.2 must be importable (``blip3d.utils.backend.setup``) before the first batch is built.
"""
from .collate import Collator, collate
from .mixture import Mixture, MixtureIterableDataset, build_mixture, load_config, task_configs
from .task import IM_VIEW_WEIGHTS, TaskConfig, ThreeDTask

__all__ = ["build_mixture", "Mixture", "MixtureIterableDataset", "load_config", "task_configs",
           "TaskConfig", "ThreeDTask", "IM_VIEW_WEIGHTS", "collate", "Collator"]

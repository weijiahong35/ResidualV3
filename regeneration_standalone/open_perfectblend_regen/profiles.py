"""Separate data, logs and pretrained drafts for each generation recipe."""
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = Path('/data/weijiahong/huggingface/models')
DEFAULT_PROFILE = 'nonthinking_ctx4096'


@dataclass(frozen=True)
class Profile:
    name: str
    thinking: str
    max_model_len: int
    draft_model: Path
    worker_suffix: str
    scheduler_name: str

    @property
    def mode(self):
        return 'thinking' if self.thinking == 'on' else 'nonthinking'

    @property
    def generation_dir(self):
        return HERE / 'pool_500k_seed42' / f'regenerated_ctx{self.max_model_len}'

    @property
    def output_root(self):
        return self.generation_dir / self.mode

    @property
    def scheduler_dir(self):
        return HERE / self.scheduler_name

    def worker_dir(self, index):
        return HERE / f'gpu{index}_{self.worker_suffix}'


PROFILES = {
    'thinking_ctx8192': Profile('thinking_ctx8192', 'on', 8192,
        MODELS / 'Qwen3-8B-speculator.dflash', 'pilot_20261004', 'auto_scheduler_20261004'),
    'nonthinking_ctx4096': Profile('nonthinking_ctx4096', 'off', 4096,
        MODELS / 'Qwen3-8B-speculator.dflash-nonthinking-20260916',
        'nonthinking_ctx4096_20261005', 'auto_scheduler_nonthinking_ctx4096_20261005'),
}


def add_profile_argument(parser):
    parser.add_argument('--profile', choices=PROFILES, default=DEFAULT_PROFILE,
                        help='Generation recipe; thinking data stays in its original directory')


def reservation_matches(pid, gpu_uuid):
    """Allow only our explicitly pinned CUDA reservation during a rolling handoff."""
    from dispatch import owned_process
    return bool(pid) and owned_process(pid, [str(HERE / 'gpu_handoff_reservation.py'),
                                            '--gpu-uuid', gpu_uuid])

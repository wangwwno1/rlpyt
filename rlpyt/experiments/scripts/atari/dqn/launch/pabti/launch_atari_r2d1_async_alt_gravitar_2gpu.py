
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/dqn/train/atari_r2d1_async_alt.py"
affinity_code = encode_affinity(
    n_cpu_core=24,
    n_gpu=4,
    async_sample=True,
    gpu_per_run=2,
    sample_gpu_per_run=2,
    # hyperthread_offset=24,
    # optim_sample_share_gpu=True,
    n_socket=1,  # Force this.
    alternating=True,
)
runs_per_setting = 1
experiment_title = "atari_r2d1_async_alt_replay"
variant_levels = list()

games = ["gravitar"]
replays = [1.35]
batch_Bs = [32]  # For 2-GPU optimization, 64/2.
values = list(zip(games, replays))
dir_names = ["{}_{}replay".format(*v) for v in values]
keys = [("env", "game"), ("algo", "replay_ratio"), ("algo", "batch_B")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "async_alt_pabti"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

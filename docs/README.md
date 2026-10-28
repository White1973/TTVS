# Quick Start

The standalone launcher initializes a local Ray head and starts the paper-aligned Qwen3-4B MATH experiment:

```bash
TTVS_LOAD_MODULES=0 \
TTVS_CONDA_ENV=ttvs \
./scripts/train_qwen3.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-4B \
  critic.model.path=/path/to/Qwen3-4B \
  trainer.default_local_dir=/path/to/checkpoints/ttvs-math
```

The final Hydra arguments override values declared earlier in the benchmark launcher. This is the recommended way to replace machine-specific model and checkpoint paths.

The wrapper accepts these environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `TTVS_CONDA_ENV` | Conda environment activated by the wrapper | `ttrl` |
| `TTVS_LOAD_MODULES` | Load the cluster Anaconda module when `1` | `1` |
| `TTVS_ANACONDA_MODULE` | Environment-module name | `Anaconda3/2023.09-0` |
| `RAY_TMPDIR` | Ray temporary directory | `<repo>/.ray_tmp` |

On a machine without environment modules, use `TTVS_LOAD_MODULES=0`. The wrapper stops an existing local Ray runtime before creating a new head node.

## Benchmark Launchers

Run a benchmark launcher directly when Ray and the environment are already initialized:

```bash
# MATH-500, 10 epochs
bash examples/ttrl/Qwen3/math.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-4B \
  critic.model.path=/path/to/Qwen3-4B

# AIME 2024, 40 epochs
bash examples/ttrl/Qwen3/aime.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-4B \
  critic.model.path=/path/to/Qwen3-4B

# AMC 2023, 20 epochs
bash examples/ttrl/Qwen3/amc.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-4B \
  critic.model.path=/path/to/Qwen3-4B

# GPQA, 10 epochs
bash examples/ttrl/Qwen3/gpqa.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-4B \
  critic.model.path=/path/to/Qwen3-4B
```

For Qwen3-8B on MATH-500, use:

```bash
bash examples/ttrl/Qwen3/math_8B.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-8B \
  critic.model.path=/path/to/Qwen3-8B
```

Compatibility launchers for Qwen2.5 and LLaMA are retained under `examples/ttrl/`. The Qwen3 launchers are the canonical path for the current `ttvs.*` implementation.

## Configuration

Important TTVS fields can be overridden from the command line:

```bash
bash examples/ttrl/Qwen3/math.sh \
  ttvs.label_vote_rollouts=32 \
  ttvs.train_rollouts_per_query=32 \
  ttvs.synthesis.generation_candidates=8 \
  ttvs.synthesis.num_variants_per_cluster=3 \
  ttvs.synthesis.ige_vote_rollouts_per_variant=32 \
  ttvs.synthesis.ige_train_rollouts_per_variant=32 \
  ttvs.cge.total_rollouts_per_cluster=32 \
  ttvs.cge.max_variants_per_cluster=3 \
  data.mixinput_step=40 \
  data.mixrollout_step=60
```

| Key | Description |
|---|---|
| `ttvs.label_vote_rollouts` | Responses used to estimate the original-query pseudo-label |
| `ttvs.train_rollouts_per_query` | Original-query rollouts retained for policy learning |
| `ttvs.synthesis.generation_candidates` | Query variants generated from an accepted seed |
| `ttvs.synthesis.num_variants_per_cluster` | Maximum selected variants associated with a semantic cluster |
| `ttvs.synthesis.ige_vote_rollouts_per_variant` | Responses used for each variant's independent IGE vote |
| `ttvs.synthesis.ige_train_rollouts_per_variant` | Variant responses retained for IGE learning |
| `ttvs.synthesis.train_synthesis_policy` | Whether query-generation trajectories are optimized; disabled by default |
| `ttvs.cge.enable` | Enables cross-group exploration |
| `ttvs.cge.total_rollouts_per_cluster` | Fixed response budget shared by a full cluster |
| `ttvs.cge.max_variants_per_cluster` | Maximum number of variants included in CGE |
| `data.mixinput_step` | Step at which online variants and IGE enter training |
| `data.mixrollout_step` | Step at which CGE begins |

`actor_rollout_ref.rollout.n` must equal `ttvs.train_rollouts_per_query`. Training rollout counts cannot exceed their corresponding voting counts.

## Monitoring

In addition to standard verl/GRPO metrics, the implementation records:

- `SvS-Valid-Ratio/Num-OQA`: accepted original-query rollouts;
- `SvS-Valid-Ratio/Num-IGE`: accepted synthesized-query IGE rollouts;
- `SvS-Valid-Ratio/Num-CGE`: accepted full-cluster CGE rollouts;
- `ttvs/rollouts/cge_total_generated`: total CGE responses generated in the step;
- `ttvs/clusters/cge_generated`: semantic clusters considered for CGE;
- `ttvs/clusters/cge_accepted`: clusters retained after online filtering;
- `ttvs/clusters/mean_variants`: mean number of selected variants per CGE cluster;
- `train/ttvs_label_accuracy`, `train/ttvs_reward_accuracy`, and related pseudo-label diagnostics.

To save extracted synthetic questions, provide:

```bash
trainer.synthetic_data_dir=/path/to/synthetic_questions
```

## Reproducing the Paper Setting

For the closest reproduction:

1. use the provided benchmark test queries as the unlabeled training stream;
2. use the Qwen3 launchers and `ppo_trainer_ttvs.yaml`;
3. keep 32 training/voting rollouts and 16 evaluation samples;
4. keep `tau_low=0.125`, `tau_high=0.875`, `Lmax=1024`;
5. enable IGE at step 40 and CGE at step 60;
6. use the paper's task-specific epoch count;
7. use eight GPUs and record both test-time training and final evaluation seeds.

The paper reports the following representative results for the Qwen3 family:

| Model | Method | MATH-500 | AIME 2024 | AMC 2023 | GPQA | Average |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B-Instruct | Initial | 59.4 | 3.3 | 28.9 | 13.1 | 26.2 |
| Qwen3-1.7B-Instruct | TTRL | 73.4 | 15.2 | 54.2 | 22.2 | 41.3 |
| Qwen3-1.7B-Instruct | TTVS | **80.2** | **23.3** | **61.4** | **26.3** | **47.5** |
| Qwen3-4B | Initial | 64.0 | 10.0 | 26.5 | 26.3 | 31.7 |
| Qwen3-4B | TTRL | 85.0 | 26.7 | 61.4 | 43.0 | 54.0 |
| Qwen3-4B | TTVS | **90.3** | **36.7** | **71.1** | **48.9** | **61.5** |
| Qwen3-8B | Initial | 82.2 | 26.9 | 57.8 | 48.0 | 53.7 |
| Qwen3-8B | TTRL | 89.2 | 46.7 | 68.6 | 53.0 | 65.4 |
| Qwen3-8B | TTVS | **92.6** | **50.0** | **72.3** | **56.1** | **67.8** |

Refer to the paper for results on Qwen2.5, LLaMA, additional datasets, ablations, computational cost, and Avg@16/Maj@16 evaluation.

## Testing

Run the focused TTVS planning tests with:

```bash
pytest -q tests/trainer/ppo/test_ttvs_utils.py
```

Run lightweight syntax checks with:

```bash
python -m py_compile \
  verl/trainer/ppo/ray_trainer.py \
  verl/trainer/ppo/ttvs_utils.py

bash -n scripts/train.sh scripts/train16.sh scripts/train_qwen3.sh
```

## Scope and Limitations

- The current reward implementation targets tasks with extractable, verifiable final answers.
- The paper's experiments are language-only; multimodal TTVS is not evaluated.
- Query synthesis quality depends on the policy's existing instruction-following and rewriting ability. The paper uses an expert-data warm-up for weaker base models.
- The included launchers contain historical machine-specific defaults. Override model and output paths from the command line.
- Full paper-scale reproduction is compute intensive: online voting, variant synthesis, IGE, and CGE all invoke rollout generation.
- Extending TTVS to subjective tasks requires a reliable model-based judge or reward model and is outside the current rule-based setup.

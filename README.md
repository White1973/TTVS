# TTVS: Test-Time Variational Synthesis

Research implementation of **TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis**, published in Findings of ACL 2026.

TTVS adapts a large reasoning model directly on unlabeled test queries. It creates its own supervision through majority voting, synthesizes semantically equivalent query variants online, and combines query-level and cluster-level exploration during reinforcement learning.

- Paper: [TTVS.pdf](https://aclanthology.org/2026.findings-acl.1634/)
- Framework: [verl](https://github.com/volcengine/verl)
- Included tasks: MATH-500, AIME 2024, AMC 2023, and GPQA
- Model families evaluated in the paper: Qwen3, Qwen2.5, and LLaMA

## Method

TTVS contains two stages.

### 1. Online Variational Synthesis

For each original unlabeled query `q`, TTVS:

1. samples multiple reasoning responses from the current policy;
2. extracts their final answers and obtains a pseudo-label by majority vote;
3. estimates query difficulty from agreement with that pseudo-label;
4. retains queries whose agreement lies within `[tau_low, tau_high]`;
5. synthesizes semantically equivalent query variants from correct reasoning trajectories;
6. filters malformed or overlength variants online.

This converts a static query into a semantic cluster:

```text
original query q
    ├── variant q'_1
    ├── variant q'_2
    └── variant q'_3
```

### 2. Test-Time Hybrid Exploration

TTVS uses two complementary policy-update groups:

- **Intra-Group Exploration (IGE):** performs majority voting and reward assignment independently for each original or synthesized query. It emphasizes accuracy on each individual formulation.
- **Cross-Group Exploration (CGE):** distributes a fixed rollout budget across the original query and its selected variants, performs one majority vote over the mixed cluster, and assigns a shared consistency reward. It encourages invariance across equivalent formulations.

```text
Unlabeled query
      │
      ▼
Pseudo-label and difficulty filtering
      │
      ▼
Online variational synthesis ──► semantic query cluster
      │                                  │
      ├──────── IGE: per-query vote ─────┤
      └──────── CGE: cluster vote ───────┘
                                         │
                                         ▼
                                  GRPO policy update
```

IGE and CGE use explicit group identities in the implementation: `query_id` identifies an IGE group, while `cluster_id` connects an original query to all of its variants for CGE.

## Installation

### Requirements

- Linux
- Python 3.10 is recommended
- CUDA-compatible NVIDIA GPUs
- Eight GPUs for the paper-scale Qwen3 launchers; smaller runs can override `trainer.n_gpus_per_node`
- A local Hugging Face model checkpoint

Create and activate an environment, then install the repository:

```bash
conda create -n ttvs python=3.10 -y
conda activate ttvs

pip install -e ".[vllm,math]"
```

## Data Preparation

Raw JSON files are included under [`data/`](data/). Each source example follows this structure:

```json
{
  "prompt": "problem text",
  "answer": "reference answer"
}
```

The reference answer is retained only for evaluation and diagnostic metrics. TTVS constructs the reward used for policy optimization from the model's own majority-vote pseudo-label.

The trainer expects Parquet files containing:

- `data_source`
- chat-formatted `prompt`
- `reward_model.ground_truth`
- `extra_info.index`

Convert a benchmark by selecting its directory in `data/preprocess.py`, then running the converter from `data/`:

```bash
cd data

# Set data_source in preprocess.py to one of:
# MATH-TTT, AIME-TTT, AMC-TTT, or GPQA-TTT
python preprocess.py

cd ..
```

The resulting files must be available as:

```text
data/<TASK>/train.parquet
data/<TASK>/test.parquet
```

## Quick Start

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

### Benchmark Launchers

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

## Citation

If you use TTVS, please cite:

```bibtex
@inproceedings{bai-etal-2026-ttvs,
  title     = {{TTVS}: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis},
  author    = {Bai, Sikai and Li, Haoxi and Zhang, Jie and Liu, Yongjiang and Guo, Song},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  month     = {July},
  pages     = {32651--32665},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgements

This repository is built on [verl](https://github.com/volcengine/verl), whose distributed training, rollout, and worker infrastructure make the TTVS implementation possible.

## License

The code is released under the [Apache License 2.0](LICENSE).

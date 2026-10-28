# TTVS: Test-Time Variational Synthesis
![TTVS](TTVS.png)
Official implementation of **TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis**, published in Findings of ACL 2026.

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
# Set data_source in preprocess.py to one of:
# MATH-TTT, AIME-TTT, AMC-TTT, or GPQA-TTT
python preprocess.py

```

The resulting files must be available as:

```text
data/<TASK>/train.parquet
data/<TASK>/test.parquet
```


### Benchmark Training

Run a benchmark launcher directly when Ray and the environment are already initialized:

```bash
bash examples/ttrl/Qwen3/math.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3-4B \
  critic.model.path=/path/to/Qwen3-4B
```

We provide more details for [`Configuration`](docs/Configuration.md).

## Citation

If you use TTVS, please cite:

```bibtex
@inproceedings{bai2026ttvs,
  title={Ttvs: Boosting self-exploring reinforcement learning via test-time variational synthesis},
  author={Bai, Sikai and Li, Haoxi and Zhang, Jie and Liu, Yongjiang and Guo, Song},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={32651--32665},
  year={2026}
}
```

## Acknowledgements

This repository is built on [verl](https://github.com/volcengine/verl), whose distributed training, rollout, and worker infrastructure make the TTVS implementation possible.

## License

The code is released under the [Apache License 2.0](LICENSE).

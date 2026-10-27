"""Pure planning utilities for Test-Time Variational Synthesis (TTVS).

The helpers in this module make the two TTVS grouping semantics explicit:

* IGE groups responses by ``query_id`` (one original or variational query).
* CGE groups responses by ``cluster_id`` (the original query and all selected
  semantic variants).

They deliberately do not depend on torch or Ray so the rollout contract can be
unit-tested without launching distributed workers.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class TTVSVariant:
    """A variational query that belongs to one TTVS semantic cluster."""

    cluster_id: Hashable
    query_id: Hashable
    prompt_index: int


@dataclass(frozen=True)
class TTVSClusterPlan:
    """Rollout allocation for one full-cluster CGE group."""

    cluster_id: Hashable
    original_prompt_index: int
    variants: tuple[TTVSVariant, ...]
    rollout_counts: tuple[int, ...]

    @property
    def query_count(self) -> int:
        return 1 + len(self.variants)

    @property
    def total_rollouts(self) -> int:
        return sum(self.rollout_counts)


def allocate_cge_rollouts(total_rollouts: int, query_count: int) -> tuple[int, ...]:
    """Evenly allocate a fixed TTVS CGE budget across a semantic cluster.

    The first entry is the original query and the remaining entries are its
    selected variants. Any remainder is assigned deterministically from the
    original query forward, which keeps runs reproducible.
    """

    if total_rollouts <= 0:
        raise ValueError("TTVS CGE total_rollouts must be positive")
    if query_count <= 0:
        raise ValueError("A TTVS cluster must contain at least one query")
    if query_count > total_rollouts:
        raise ValueError(
            "TTVS CGE requires at least one rollout per query: "
            f"query_count={query_count}, total_rollouts={total_rollouts}"
        )

    quotient, remainder = divmod(total_rollouts, query_count)
    return tuple(quotient + (index < remainder) for index in range(query_count))


def build_full_cluster_plans(
    original_cluster_ids: Sequence[Hashable],
    variants: Iterable[TTVSVariant],
    *,
    total_rollouts: int,
    max_variants_per_cluster: int,
) -> list[TTVSClusterPlan]:
    """Build stable full-cluster CGE plans for TTVS.

    Duplicate ``query_id`` values are removed while preserving their first
    occurrence. Clusters without a valid variant are omitted because they do
    not provide cross-query exploration and remain available to IGE.
    """

    if max_variants_per_cluster <= 0:
        raise ValueError("TTVS max_variants_per_cluster must be positive")

    original_index = {}
    for index, cluster_id in enumerate(original_cluster_ids):
        original_index.setdefault(cluster_id, index)

    variants_by_cluster: dict[Hashable, list[TTVSVariant]] = {}
    seen_queries: dict[Hashable, set[Hashable]] = {}
    for variant in variants:
        if variant.cluster_id not in original_index:
            continue
        seen = seen_queries.setdefault(variant.cluster_id, set())
        if variant.query_id in seen:
            continue
        seen.add(variant.query_id)
        variants_by_cluster.setdefault(variant.cluster_id, []).append(variant)

    plans = []
    for cluster_id in original_cluster_ids:
        selected = tuple(variants_by_cluster.get(cluster_id, ())[:max_variants_per_cluster])
        if not selected:
            continue
        counts = allocate_cge_rollouts(total_rollouts, 1 + len(selected))
        plans.append(
            TTVSClusterPlan(
                cluster_id=cluster_id,
                original_prompt_index=original_index[cluster_id],
                variants=selected,
                rollout_counts=counts,
            )
        )
    return plans


def apply_ttvs_majority_gt(batch, generated, group_size: int, tokenizer):
    """Attach one self-generated TTVS pseudo-label to every query group.

    The dataset ground truth is preserved while IGE and full-cluster CGE
    replace the active reward target independently.
    """

    from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string

    if group_size <= 0 or len(generated) % group_size:
        raise ValueError("Invalid TTVS majority-vote group size")
    if len(batch) != len(generated) // group_size:
        raise ValueError("TTVS pseudo-label batch does not match generated groups")

    outputs = []
    for item in generated:
        prompt_length = item.batch["prompts"].shape[-1]
        valid_length = item.batch["attention_mask"][prompt_length:].sum()
        outputs.append(tokenizer.decode(item.batch["responses"][:valid_length], skip_special_tokens=True))

    labels = []
    ratios = []
    for start in range(0, len(outputs), group_size):
        answers = [extract_answer(output) for output in outputs[start : start + group_size]]
        answers = [simplify_expression_string(answer) for answer in answers if answer is not None]
        if not answers:
            labels.append("None")
            ratios.append(0.0)
            continue
        label, count = Counter(answers).most_common(1)[0]
        labels.append(label)
        ratios.append(count / group_size)
    for index, label in enumerate(labels):
        reward_model = batch[index].non_tensor_batch["reward_model"]
        dataset_gt = reward_model.get("original_gt", reward_model["ground_truth"])
        reward_model["ground_truth"] = label
        reward_model["majority_gt"] = label
        reward_model["original_gt"] = dataset_gt
    batch.non_tensor_batch["majority_ratio_list"] = np.asarray(ratios, dtype=float)
    return batch


def select_ttvs_rollouts(generated, vote_rollouts: int, train_rollouts: int):
    """Sample the configured TTVS training rollouts from each vote group."""

    if vote_rollouts <= 0 or train_rollouts <= 0 or train_rollouts > vote_rollouts:
        raise ValueError("Invalid TTVS vote/training rollout counts")
    if len(generated) % vote_rollouts:
        raise ValueError("Generated TTVS rollouts do not form complete vote groups")

    selected = []
    for start in range(0, len(generated), vote_rollouts):
        selected.extend(random.sample(range(start, start + vote_rollouts), train_rollouts))
    return generated[selected]


def restore_ttvs_original_gt(batch):
    """Restore dataset labels after a TTVS policy update for evaluation."""

    for item in batch:
        reward_model = item.non_tensor_batch["reward_model"]
        reward_model["ground_truth"] = reward_model["original_gt"]
    return batch


def compute_ttvs_metrics(batch) -> dict[str, float]:
    """Compute TTVS pseudo-label metrics using semantic rollout groups.

    TTVS batches can contain OQA, IGE, and CGE groups with different rollout
    counts. Grouping by ``uid`` avoids conflating these objectives when they
    originate from the same dataset row.
    """

    from verl.utils.reward_score.ttrl_math import grade

    groups: dict[Hashable, list[int]] = {}
    for index, uid in enumerate(batch.non_tensor_batch["uid"]):
        groups.setdefault(uid, []).append(index)
    if not groups:
        return {}

    group_metrics = []
    for indices in groups.values():
        majority_labels = {
            batch[index].non_tensor_batch["reward_model"]["majority_gt"] for index in indices
        }
        original_labels = {
            batch[index].non_tensor_batch["reward_model"]["original_gt"] for index in indices
        }
        if len(majority_labels) != 1 or len(original_labels) != 1:
            raise ValueError("A TTVS uid group contains inconsistent reward labels")

        majority_rewards = [
            batch[index].batch["token_level_scores"].sum().item() for index in indices
        ]
        original_rewards = [
            batch[index].batch["token_level_scores_original"].sum().item() for index in indices
        ]
        majority_label = next(iter(majority_labels))
        original_label = next(iter(original_labels))
        ratio = float(batch.non_tensor_batch["majority_ratio_list"][indices[0]])
        group_metrics.append(
            {
                "label_accuracy": float(grade(majority_label, original_label)),
                "reward_accuracy": float(
                    np.mean(
                        [predicted == reference for predicted, reference in zip(majority_rewards, original_rewards)]
                    )
                ),
                "majority_voting_reward": float(np.mean(majority_rewards)),
                "ground_truth_reward": float(np.mean(original_rewards)),
                "pass_any_ground_truth": float(sum(original_rewards) >= 1),
                "majority_ratio": ratio,
                "rollouts_per_group": float(len(indices)),
            }
        )

    return {
        key: float(np.mean([metrics[key] for metrics in group_metrics]))
        for key in group_metrics[0]
    }

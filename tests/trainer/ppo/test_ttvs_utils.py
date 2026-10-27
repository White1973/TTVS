import pytest

from verl.trainer.ppo.ttvs_utils import (
    TTVSVariant,
    allocate_cge_rollouts,
    build_full_cluster_plans,
)


def test_allocate_cge_rollouts_even_cluster():
    assert allocate_cge_rollouts(32, 4) == (8, 8, 8, 8)


def test_allocate_cge_rollouts_distributes_remainder_stably():
    assert allocate_cge_rollouts(32, 5) == (7, 7, 6, 6, 6)


def test_full_cluster_plan_keeps_all_selected_variants():
    variants = [
        TTVSVariant("a", "a-v0", 10),
        TTVSVariant("a", "a-v1", 11),
        TTVSVariant("a", "a-v2", 12),
        TTVSVariant("b", "b-v0", 20),
    ]
    plans = build_full_cluster_plans(
        ["a", "b"],
        variants,
        total_rollouts=32,
        max_variants_per_cluster=3,
    )

    assert [variant.query_id for variant in plans[0].variants] == ["a-v0", "a-v1", "a-v2"]
    assert plans[0].rollout_counts == (8, 8, 8, 8)
    assert plans[0].total_rollouts == 32
    assert plans[1].rollout_counts == (16, 16)


def test_full_cluster_plan_deduplicates_query_ids_and_omits_empty_clusters():
    plans = build_full_cluster_plans(
        ["a", "b"],
        [TTVSVariant("a", "a-v0", 3), TTVSVariant("a", "a-v0", 4)],
        total_rollouts=8,
        max_variants_per_cluster=3,
    )

    assert len(plans) == 1
    assert plans[0].cluster_id == "a"
    assert plans[0].variants[0].prompt_index == 3


@pytest.mark.parametrize("total, queries", [(0, 1), (2, 3)])
def test_allocate_cge_rollouts_rejects_invalid_budget(total, queries):
    with pytest.raises(ValueError):
        allocate_cge_rollouts(total, queries)


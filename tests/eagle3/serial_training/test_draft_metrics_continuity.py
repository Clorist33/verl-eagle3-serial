# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""draft 指标在未训练的步也要有值，否则曲线会断。

k>1 时 draft 只在 global_steps % k == 0 的步训练（k=5 时只有 step 5/10/15...），
其余步 metrics 里根本没有 draft/* 键，指标行时有时无。实测
eagle3_v3_8b_100s_k5_t10_spec3_20260901_114654.log：
    step 1/2/3/4 无 draft_loss，step 5 有，step 6/7/8/9 无，step 10 有 ...

这里验证补 0 的行为，以及两条不能破的约束：
  ① 补的键名必须与 worker 侧产出的一致，否则会分裂成两条曲线；
  ② 不能覆盖真实值。
"""

import inspect
import re


def _trainer():
    """拿到未实例化的 trainer 类 —— 只测这两个纯函数，不需要真实 trainer。"""
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    return PPOTrainer


def test_keys_match_worker_side():
    """补 0 用的键名必须与 engine_workers 产出的完全一致。

    对不上会导致：真实训练步产出 draft/draft_loss，补 0 的步产出别的名字，
    TensorBoard 里成两条曲线，比不出趋势。
    """
    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.update_draft_deferred)
    # worker 侧 metrics dict 里的键（形如 "draft_loss": [...]）
    worker_keys = set(re.findall(r'"(draft_[a-z_]+)":', src))
    assert worker_keys, "没能从 worker 源码里解析出 draft 指标键，测试假设已失效"

    filled = set(_trainer()._DRAFT_METRIC_KEYS)
    assert filled == worker_keys, (
        f"键名不一致 —— 只在 worker 侧: {worker_keys - filled}；"
        f"只在补 0 侧: {filled - worker_keys}"
    )


def test_fills_zero_for_all_keys():
    metrics = {}
    _trainer()._fill_draft_metrics_when_skipped(_trainer(), metrics)

    for key in _trainer()._DRAFT_METRIC_KEYS:
        assert metrics[f"draft/{key}"] == 0.0, f"draft/{key} 未被补 0"


def test_does_not_overwrite_real_values():
    """真实训练步已写入的值不能被覆盖成 0。"""
    metrics = {
        "draft/draft_loss": 1.234,
        "draft/draft_updates": 10.0,
    }
    _trainer()._fill_draft_metrics_when_skipped(_trainer(), metrics)

    assert metrics["draft/draft_loss"] == 1.234, "真实 loss 被覆盖了"
    assert metrics["draft/draft_updates"] == 10.0, "真实 updates 被覆盖了"
    # 其余未写入的键仍应补 0
    assert metrics["draft/draft_windows"] == 0.0


def test_values_are_plain_floats_not_lists():
    """必须是标量。

    worker 侧产出的是 list（[0.1]），要经 reduce_metrics 才变标量。补 0 这条路
    绕过了 reduce_metrics，所以必须直接给 float —— 给 list 会被
    aggregate_logger.py 的 isinstance(v, numbers.Number) 静默丢弃，
    TensorBoard 的 add_scalar 还会抛异常。
    """
    import numbers

    metrics = {}
    _trainer()._fill_draft_metrics_when_skipped(_trainer(), metrics)

    for key, val in metrics.items():
        assert isinstance(val, numbers.Number), f"{key} 不是标量而是 {type(val).__name__}"
        assert not isinstance(val, (list, tuple)), f"{key} 是序列，会被 logger 丢弃"


def test_skip_branch_calls_the_filler():
    """静态守卫：_step_once_serial 的 else 分支必须调用补 0，否则曲线又会断。"""
    src = inspect.getsource(_trainer()._step_once_serial)
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))

    assert "_fill_draft_metrics_when_skipped" in code, (
        "未训练 draft 的分支没有补指标，k>1 时曲线会断断续续"
    )


def test_failed_training_also_filled():
    """本该训练却没拿到指标（worker 返回 None）时也要补 0。

    否则"应训却失败"的步会留空洞，和"按 k 跳过"的步分不清。
    """
    src = inspect.getsource(_trainer()._update_draft_deferred)
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))

    assert "_fill_draft_metrics_when_skipped" in code, (
        "output is None 的早退路径没有补指标"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

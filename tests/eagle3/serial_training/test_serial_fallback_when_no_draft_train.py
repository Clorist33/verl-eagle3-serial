"""enable_train=False 时串行路径必须回落到并行（原版 PPO）主干。

背景：两个开关是独立的 —— algorithm.eagle3.enable_serial_training 选模式，
model.eagle3.enable_train 决定 draft 是否训练。后者为 False 时 draft 压根没被
构建（engine._eagle3 恒为 None），串行路径里三处 draft 调用全部空转，还会每 k 步
刷一条误导性警告。所以要回落。

eagle3 推理由 model.eagle3.enable_rollout 独立控制，本回落不能影响它。
"""

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

TRAINER_BASE = (
    Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "v1" / "trainer_base.py"
)


def _predicate_source() -> str:
    """取 _is_serial_training_enabled 的源码（不 import trainer_base，避免拖入 ray/megatron）。"""
    tree = ast.parse(TRAINER_BASE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_is_serial_training_enabled":
            return ast.get_source_segment(TRAINER_BASE.read_text(), node)
    raise AssertionError("_is_serial_training_enabled 不在 trainer_base.py 里了")


def _run_predicate(serial: bool, enable_train, has_eagle3_key: bool = True):
    """把方法源码单独编译出来在假 self 上执行，隔离 trainer 的重量级依赖。"""
    src = _predicate_source()
    model = {"path": "/dev/null"}
    if has_eagle3_key:
        model["eagle3"] = {"enable_train": enable_train, "enable_rollout": True}
    cfg = OmegaConf.create(
        {
            "algorithm": {"eagle3": {"enable_serial_training": serial}},
            "actor_rollout_ref": {"model": model},
        }
    )

    warnings: list[str] = []

    class _Logger:
        def warning(self, msg, *a):
            warnings.append(msg % a if a else msg)

    # get_source_segment 的 def 行无缩进、函数体带 8 空格，没有公共前缀可 dedent。
    # 整段重新缩进后塞进壳类，缩进就自洽了。
    body = "\n".join("    " + line if line.strip() else line for line in src.splitlines())
    ns = {"logger": _Logger()}
    exec(f"class _Holder:\n{body}\n", ns)

    class _Fake:
        config = cfg

    return ns["_Holder"]._is_serial_training_enabled(_Fake()), warnings


@pytest.mark.parametrize(
    "serial,enable_train,expected",
    [
        (True, True, True),    # 正常串行训练
        (True, False, False),  # 本次修复：回落到并行主干
        (False, True, False),  # 明确选并行
        (False, False, False),
    ],
)
def test_both_switches_required(serial, enable_train, expected):
    got, _ = _run_predicate(serial, enable_train)
    assert got is expected, (
        f"enable_serial_training={serial} enable_train={enable_train} "
        f"应返回 {expected}，实际 {got}"
    )


def test_fallback_emits_warning():
    """回落必须留痕，否则日志里看不出为什么没走串行。"""
    got, warnings = _run_predicate(True, False)
    assert got is False
    assert any("enable_train=False" in w for w in warnings), f"没有回落警告: {warnings}"


def test_no_warning_on_normal_serial():
    _, warnings = _run_predicate(True, True)
    assert warnings == [], f"正常串行不该有回落警告: {warnings}"


def test_missing_eagle3_key_falls_back():
    """model 下整块没有 eagle3 时不能 KeyError，按不训 draft 处理。"""
    got, _ = _run_predicate(True, None, has_eagle3_key=False)
    assert got is False


def test_revert_guard():
    """反向验证：改回只看 enable_serial_training 的旧实现，回落用例必须失败。

    防止本测试因为断言写得太松而对回归无感。
    """
    old = (
        "def _is_serial_training_enabled(self):\n"
        "    eagle3_config = self.config.algorithm.get('eagle3', {})\n"
        "    return bool(eagle3_config.get('enable_serial_training', False))\n"
    )
    cfg = OmegaConf.create(
        {
            "algorithm": {"eagle3": {"enable_serial_training": True}},
            "actor_rollout_ref": {"model": {"eagle3": {"enable_train": False}}},
        }
    )
    ns: dict = {}
    exec(old, ns)

    class _Fake:
        config = cfg

    assert ns["_is_serial_training_enabled"](_Fake()) is True, (
        "旧实现在 enable_train=False 下返回 True —— 这正是本次要修的行为；"
        "若这里为 False 说明测试构造没打到点上"
    )

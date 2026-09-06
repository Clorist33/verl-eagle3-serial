# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import logging
import math
import os
import sys
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import StringIO
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from packaging.version import InvalidVersion, Version
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    RolloutMoELoadBalanceMetricsAccumulator,
    compute_data_metrics,
    compute_moe_lb_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    get_metric_data_with_optional_routed_experts,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_draft_metrics, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
)
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY, ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator, compute_advantage_for_multi_trajectories
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_type
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip import SkipManager
from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _tq_supports_checkpoint() -> bool:
    """Whether the installed TransferQueue can snapshot/restore its state for checkpoint consistency."""
    try:
        version_supported = Version(getattr(tq, "__version__", "")) >= Version("0.1.9")
    except InvalidVersion:
        return False
    return (
        version_supported
        and callable(getattr(tq, "save_checkpoint", None))
        and callable(getattr(tq, "load_checkpoint", None))
    )


class PPOTrainer(ABC):
    """Base class for PPO trainer.

    Args:
        config: DictConfig from yaml config file.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        self.replay_buffer = self._build_replay_buffer()
        self._rollout_moe_lb_metrics_accumulator = RolloutMoELoadBalanceMetricsAccumulator(
            model_config=self.config.actor_rollout_ref.model
        )

    def _build_replay_buffer(self) -> ReplayBuffer:
        """Instantiate the replay buffer (or a user-provided custom sampler).

        Set ``trainer.v1.sampler.custom_sampler.{path,name}`` to plug in a custom
        ``ReplayBuffer`` subclass; otherwise the built-in implementation is used.
        """
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        has_custom_sampler = bool(
            custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name")
        )
        if has_custom_sampler:
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)
        else:
            sampler_cls = ReplayBuffer if self.trainer_mode == "sync" else ReplayBufferAsync

        replay_buffer_kwargs = dict(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
        )
        # Preserve the existing constructor contract for external samplers; custom implementations own
        # their filtering semantics and can consume algorithm.filter_groups through their own config.
        if not has_custom_sampler:
            filter_groups_metric = self._resolve_filter_groups_metric()
            sync_refill_failed_groups = bool(sampler_config.get("sync_refill_failed_groups", False))
            replay_buffer_kwargs.update(
                filter_groups_metric=filter_groups_metric,
                sync_refill_failed_groups=sync_refill_failed_groups,
            )
            if sampler_cls is ReplayBuffer:
                filter_groups = self.config.algorithm.get("filter_groups", None)
                max_inflight_gen_batches = 1
                if filter_groups_metric is not None:
                    max_inflight_gen_batches = filter_groups.get("max_inflight_gen_batches", 1)
                train_batch_size = self.config.data.train_batch_size
                replay_buffer_kwargs.update(
                    train_batch_size=train_batch_size,
                    gen_batch_size=1
                    if filter_groups_metric is not None or sync_refill_failed_groups
                    else (self.config.data.get("gen_batch_size", None) or train_batch_size),
                    max_inflight_gen_batches=max_inflight_gen_batches,
                )
        return sampler_cls(**replay_buffer_kwargs)

    def _resolve_filter_groups_metric(self) -> str | None:
        """Resolve DAPO's group metric and verify that rollout computes it before sampling."""
        filter_groups = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        if not filter_enabled:
            return None

        filter_metric = filter_groups.get("metric", None)
        if not filter_metric:
            raise ValueError("algorithm.filter_groups.metric must be set when group filtering is enabled")

        reward_model = self.config.reward.reward_model
        streaming_reward_path = not reward_model.enable or reward_model.enable_resource_pool
        assert streaming_reward_path, (
            "algorithm.filter_groups requires the reward metric at sampling time: use rule-based reward or "
            "reward.reward_model.enable_resource_pool=True. A colocated reward model computes rewards only "
            "after replay-buffer sampling."
        )
        max_num_gen_batches = filter_groups.get("max_num_gen_batches", 0)
        if max_num_gen_batches > 0:
            logger.warning(
                "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
                "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
                max_num_gen_batches,
            )
        return str(filter_metric)

    def init(self):
        """Initialize all components of the trainer.

        1. WorkerGroup: actor, critic, reference with model engine: FSDP/Megatron/VeOmni/...
        2. LLMServerManager: launch and manage LLM server replicas for generation.
        3. CheckpointEngineManager: sync weights between worker group and LLM server replicas.
        4. RewardLoopManager: reward workers for rule-based reward, optional LLM server for model-based reward.
        5. [Optional] MultiTeacherModelManager: LLM teacher servers for on-policy distillation.
        """
        self._setup()
        self.on_init_end()

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu

            # Wire the critic profiler config via the hydra path (real dataclass tool_config), so the
            # standalone critic TrainingWorker gets a working DistProfiler instead of a silent no-op.
            critic_omega_profiler_config = self.config.critic.get("profiler", {})
            critic_profiler_config = (
                omega_conf_to_dataclass(critic_omega_profiler_config) if critic_omega_profiler_config else None
            )

            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
                profiler_config=critic_profiler_config,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )                # 创建 16 个 ActorRolloutRefWorker 实例，每个 worker 在创建时跑 __init__，但 __init__ 只建 engine 对象，不初始化
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()                  # Driver 进程 往 16 个 worker 进程发消息。RPC 调所有 worker 的 init_model，会跳到verl/workers/engine_workers.py 的 init_model()
                                                            # Driver 进程（主进程），self.actor_rollout_wg.init_model()这句指令在主进程
                                                            # │  通过 Ray 框架，发送 "请执行 init_model()" 消息
                                                            # ├──► Worker 0 进程（GPU 0）
                                                            # ├──► Worker 1 进程（GPU 1）
                                                            # ├──► ...
                                                            # └──► Worker 15 进程（GPU 15）
                                                            #  RPC = Remote Procedure Call（远程过程调用）简单说：在进程 A 里调用一个函数，但这个函数真正在进程 B 里执行。
                                                            # actor_rollout_wg 是一个 RayWorkerGroup，它代表 16 个 worker 的集合。调它的方法 = RPC 到所有 worker。
        

        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(actor_role)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        self.llm_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_engine_config.backend = "naive"
        self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("all initialize finished, ready to fit")

    def get_llm_client(self) -> LLMServerClient:
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client()

    def get_teacher_client(self) -> Optional[dict[str, LLMServerClient]]:
        """Get the On-Policy Distillation teacher server clients.

        Returns:
            dict[str, LLMServerClient]: The teacher server clients.
        """
        return self.teacher_model_manager.get_client() if self.use_teacher_policy else None

    def get_reward_handles(self) -> list[ray.actor.ActorHandle]:
        """Get the handles of reward loop workers."""
        return self.reward_loop_manager.reward_loop_worker_handles

    def fit(self, agent_loop_manager: AgentLoopManager):
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager

        # === 初始化串行训练配置（如果启用）===
        if self._is_serial_training_enabled():
            self._initialize_serial_training_config()

        # initialize SkipManager for V1 rollout skip support
        SkipManager.init(self.config)

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            val_metrics = self._validate()
            self.on_validate_end()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // self.steps_per_epoch

        # ===创建进度条（只执行1次，每个step只输出一行到日志，会输出初始状态 0/200 [00:00<?, ?it/s] ）==
        # 创建StringIO buffer来捕获tqdm的输出
        self.tqdm_buffer = StringIO()

        # 检测是否是终端运行
        is_terminal = sys.stdout.isatty()

        # 如果是终端，同时输出到stdout（实时显示）和buffer（记录）
        # 如果不是终端（重定向到文件），只输出到buffer
        if is_terminal:
            # 定义一个简单的Tee类，同时写入多个目标
            class TeeFile:
                def __init__(self, *files):
                    self.files = files
                def write(self, data):
                    for f in self.files:
                        f.write(data)
                def flush(self):
                    for f in self.files:
                        f.flush()
            tqdm_output = TeeFile(sys.stdout, self.tqdm_buffer)
        else:
            tqdm_output = self.tqdm_buffer

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            file=tqdm_output
        )

        # we start from step 1
        self.global_steps += 1
        # SkipManager skips warmup batches in async trainers, so it doesn't conflict with reissue.
        SkipManager.set_step(self.global_steps)
        self._reissue_inflight_prompts()
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        self.on_train_begin()
        last_val_metrics = None
        while current_epoch < self.config.trainer.total_epochs and self.global_steps <= self.total_training_steps:
            is_last_step = self.global_steps >= self.total_training_steps
            metrics = {}
            self.timing_raw = {}

            # 1. perform rollout and actor/critic training
            with marked_timer("step", self.timing_raw):
                self.on_step_begin()

                self._start_profiling()
                batch = self.step(metrics, self.timing_raw)     # 开始进入前向步
                self._stop_profiling()

                # 2. save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                        self._save_checkpoint()

                self.on_step_end()
                metrics.update(self._consume_sync_metrics())

            # 4. validate
            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    val_metrics: dict = self._validate()
                    self.on_validate_end()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            # 5. record metrics
            self._compute_metrics(batch, metrics, self.timing_raw, global_steps=self.global_steps, epoch=current_epoch)

            # 6. dump rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, self.timing_raw, rollout_data_dir)

            # 7. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

            dapo_filtered_reward_counts = metrics.pop(DAPO_FILTERED_REWARD_COUNTS_KEY, None)
            self.logger.log(data=metrics, step=self.global_steps)
            if dapo_filtered_reward_counts:
                self.dapo_filtered_reward_logger.log(
                    self.config.trainer.logger, dapo_filtered_reward_counts, self.global_steps
                )

            # === 更新进度条，显示当前步数和训练类型 ===
            if hasattr(self, '_current_training_type'):
                # 串行训练模式：显示 global_steps 和 actor_steps/draft_steps
                progress_desc = (
                    f"Global {self.global_steps}/{self.total_training_steps} "
                    f"[Actor {self.actor_steps}/{self.actor_training_steps}]"
                )
                if self._current_training_type == "Actor+Draft":
                    progress_desc += f" [Draft {self.draft_steps}/{self.draft_training_steps}]"
            else:
                # 并行训练模式：只显示步数
                progress_desc = f"Step {self.global_steps}"

            progress_bar.set_description(progress_desc)
            progress_bar.update(1)

            # === 在非终端模式（日志文件）下，只输出最后一行状态 ===
            if not is_terminal:
                # 获取buffer中的所有内容
                buffer_content = self.tqdm_buffer.getvalue()
                # tqdm可能使用\r或\n作为分隔符，需要同时处理
                # 先把\r替换成\n，然后统一按\n分割
                buffer_content = buffer_content.replace('\r', '\n')
                # 只取最后一行（最新的进度条状态）
                lines = buffer_content.strip().split('\n')
                if lines and lines[-1]:
                    # 输出到真实stdout（会进入日志文件），保留进度条视觉效果
                    print(lines[-1], flush=True)
                # 清空buffer，准备下一个step
                self.tqdm_buffer.truncate(0)
                self.tqdm_buffer.seek(0)

            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

        self.on_train_end()
        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        train_batch_size = self.config.data.train_batch_size
        assert train_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({train_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )   # self.parameter_sync_step 代表？？？？？？？？？？？？？？？？？？？？？
        sample_batch_size = train_batch_size // self.parameter_sync_step

        self._add_batch_to_generate()  # 进行推理

        metrics_aggregator = MetricsAggregator()
        combined_keys: list = []
        combined_tags: list = []
        combined_partition_id = "train"
        for _ in range(self.parameter_sync_step):
            iter_metrics: dict = {}
            batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)    # 进行训练
            sample_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
            metrics_aggregator.add_step_metrics(iter_metrics, sample_count=sample_count)
            combined_keys.extend(batch.keys)
            combined_tags.extend(batch.tags)
            combined_partition_id = batch.partition_id

        metrics.update(metrics_aggregator.get_aggregated_metrics())
        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """Run a single local update: sample one mini-batch and perform the full PPO pipeline once.

        【路由方法】根据配置决定使用串行模式还是并行模式。
        本方法不包含任何业务逻辑，只做路由判断。
        """
        # ========== 路由：串行 or 并行 ==========
        if self._is_serial_training_enabled():
            # 串行训练路径（新增）
            return self._step_once_serial(metrics, timing_raw, sample_batch_size)
        else:
            # 并行训练路径（原有逻辑，封装后）
            return self._step_once_parallel(metrics, timing_raw, sample_batch_size)

    def _is_serial_training_enabled(self) -> bool:
        """判断是否启用串行训练。

        两个开关都要为真才走串行路径：

        - ``algorithm.eagle3.enable_serial_training``：选择串行而非并行模式；
        - ``actor_rollout_ref.model.eagle3.enable_train``：draft 是否参与训练。

        为什么后者也要看：``enable_train=False`` 时 draft 压根没被构建
        （megatron/transformer_impl.py 的 initialize 里 ``if _e3 is not None and
        _e3.enable_train`` 不成立 → ``engine._eagle3`` 恒为 None），于是串行路径里
        那三处 draft 调用全部空转 —— 采集闸门 ``eagle3_collect_only`` 开着也抓不到
        东西（hidden capture 的 hook 随 setup 一起没装）、``snapshot_draft_teacher``
        和 ``update_draft_deferred`` 各自在 worker 侧 early-return。结果是白付两次
        跨进程 RPC，并且每 k 步刷一条 "本步应训 draft，但 worker 没有返回指标" 的
        警告 —— 那条话术假定"本该训练却失败了"，而实际是配置上就没开，排查时会
        误导人往采集长度门的方向查。

        所以这里回落到 ``_step_once_parallel``。注意它并不是"并行训 draft"的路径，
        而是 verl 原有的 PPO 主干（整段不含任何 draft/eagle3 调用）；并行模式下
        draft 是靠 policy 前向里的 hook 顺带训的，hook 同样由 setup 安装。因此
        ``enable_train=False`` 下走这条分支，draft 一样不会训，只是更干净。

        eagle3 参与推理不受影响：drafter 在 rollout 侧由
        ``model.eagle3.enable_rollout`` 独立控制，与本判断无关。

        三个调用点（初始化、路由、metrics）共用本方法，一起回落才自洽：
        ``_initialize_serial_training_config`` 不执行 → ``actor_steps`` /
        ``draft_steps`` 等属性不存在 → 串行专用 metrics 也必须同步跳过，否则
        AttributeError。
        """
        eagle3_config = self.config.algorithm.get('eagle3', {})
        if not eagle3_config.get('enable_serial_training', False):
            return False

        model_eagle3 = self.config.actor_rollout_ref.model.get('eagle3', None)
        enable_train = bool(model_eagle3 is not None and model_eagle3.get('enable_train', False))
        if not enable_train:
            logger.warning(
                "[Serial Training] enable_serial_training=True 但 model.eagle3.enable_train=False："
                "draft 不参与训练，回落到并行（原版 PPO）主干。eagle3 推理不受影响"
                "（由 model.eagle3.enable_rollout 控制）。"
            )
        return enable_train

    def _initialize_serial_training_config(self):
        """初始化串行训练配置（参数验证已在启动前的 validate_config 完成）

        本方法只负责：
        1. 读取已验证的配置参数
        2. 存储到 self
        3. 初始化步数计数器
        4. 输出初始化日志
        """
        # 1. 获取配置参数（已在 validate_config 验证过，这里直接读取）
        actor_training_steps = self.config.trainer.get('actor_training_steps')
        k = self.config.algorithm.eagle3.get('actor_steps_per_draft_step', 5)

        # 2. 计算推导参数
        # v3：draft 不再占用独立的 global step，它搭 actor 步的车。所以
        # total == actor，draft_training_steps 只是「draft 会被训练多少次」，
        # 不是「多少个步」。v1/v2 的 total = actor + draft 在这里不再成立。
        draft_training_steps = actor_training_steps // k
        total_training_steps = actor_training_steps

        # 3. 存储参数
        self.actor_training_steps = actor_training_steps
        self.draft_training_steps = draft_training_steps
        self.total_training_steps = total_training_steps

        # 4. 初始化步数计数器
        self.actor_steps = 0  # Actor 实际完成的训练步数
        self.draft_steps = 0  # Draft 实际完成的训练步数
        # self.global_steps 已在父类初始化

        # 5. 日志输出
        # period = k + 1
        # num_cycles = total_training_steps // period
        # logger.info("=" * 60)
        # logger.info("[Serial Training] Initialized scheduler:")
        # logger.info(f"  actor_training_steps:       {self.actor_training_steps}")
        # logger.info(f"  actor_steps_per_draft_step: {k}")
        # logger.info(f"  draft_training_steps:       {self.draft_training_steps}")
        # logger.info(f"  total_training_steps:       {self.total_training_steps}")
        # logger.info(f"  training_ratio:             Actor:{self.actor_training_steps} / Draft:{self.draft_training_steps} = {k}:1")
        # logger.info(f"  period (k+1):               {period} steps/cycle")
        # logger.info(f"  num_cycles:                 {num_cycles} complete cycles")
        # logger.info("=" * 60)

    def _step_once_parallel(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """并行训练流程（原有逻辑）：Actor 和 Draft 同时训练"""
        # 1. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        # Check if rollout_only mode is enabled
        rollout_only = self.config.trainer.get("rollout_only", False)

        if rollout_only:
            # Rollout-only mode: skip all training updates
            logger.info(f"[Rollout-Only Mode] Step {self.global_steps}: Skipping training updates, only performing rollout")
            # Still compute basic metrics for monitoring
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                with marked_timer("reward", timing_raw, color="yellow"):
                    batch = self._compute_reward_colocate(batch, metrics=metrics)

            # Add dummy training fields to TransferQueue to avoid KeyError in metrics computation
            # These are required by compute_data_metrics() but not computed in rollout_only mode
            import torch
            # Read response_mask from TransferQueue to get the shape
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["response_mask"])
            response_mask = data["response_mask"]

            # Create dummy advantages and returns with the same structure as response_mask (nested tensor)
            dummy_advantages = torch.nested.as_nested_tensor(
                [torch.zeros(len(response_mask[i]), dtype=torch.float32, device=response_mask.device)
                 for i in range(len(batch))],
                layout=torch.jagged,
            )
            dummy_returns = torch.nested.as_nested_tensor(
                [torch.zeros(len(response_mask[i]), dtype=torch.float32, device=response_mask.device)
                 for i in range(len(batch))],
                layout=torch.jagged,
            )

            # Write dummy fields back to TransferQueue
            tq.kv_batch_put(
                keys=batch.keys,
                partition_id=batch.partition_id,
                fields=tu.get_tensordict({"advantages": dummy_advantages, "returns": dummy_returns}),
            )

            return batch

        # Normal training mode: perform full PPO pipeline
        # 2. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 3. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 4. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 5. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 7. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 8. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 9. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    def _step_once_serial(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """串行训练流程：Actor 和 Draft 交替训练（新增方法）

        【新增逻辑】与原有逻辑完全独立，不影响并行模式。
        """
        # 1. 初始化调度器（首次调用）
        if not hasattr(self, '_serial_scheduler'):
            eagle3_config = self.config.algorithm.get('eagle3', {})
            k = eagle3_config.get('actor_steps_per_draft_step', 5)
            self._serial_scheduler = SerialTrainingScheduler(k)
            logger.info(f"[Serial Training] Initialized scheduler with k={k}")

        # 2. 判断当前步骤类型
        train_actor = self._serial_scheduler.should_train_actor(self.global_steps)  # 恒为true
        train_draft = self._serial_scheduler.should_train_draft(self.global_steps)  # 第k步才是ture

        # === 记录当前步骤类型（用于进度条显示）===
        # v3 没有「Draft 步」了：每一步都是 Actor 步，第 k 步额外带上 draft。
        # 旧的三分支里 "Draft"/"Unknown" 现在都不可达（should_train_actor 恒 True）。
        self._current_training_type = "Actor+Draft" if train_draft else "Actor"

        logger.debug(
            f"[Serial Training] Step {self.global_steps}: train_actor={train_actor}, train_draft={train_draft}"
        )

        # 3. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        print("*"*100)
        print("*"*100)
        print("replay buffer跑完"*100)
        print("*"*100)
        print("*"*100)

        # 4. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 5. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        print("*"*100)
        print("*"*100)
        print("balance batch跑完"*100)
        print("*"*100)
        print("*"*100)

        # ========== v3：只有一种步，draft 每 k 步搭一次车 ==========
        # v1/v2 在这里分流成「Actor 步」和「Draft 步」，而 rollout 在 :790 早已跑完，
        # 于是 Draft 步白白付掉一整次推理（实测占该步 89% 时间）。v3 取消分流：每步都是
        # 完整的 Actor 步，draft 只是在需要的那些步上多做两件事 —— 在 old_log_prob 那次
        # 前向里采特征，在 update_actor 之后用采到的特征训练。两件事都不需要额外前向。
        if True:
            # 注意：extra_info 会被 _compute_old_log_prob 内的 tq.kv_batch_put 整体重建
            # （返回全新 KVBatchMeta，extra_info 为空），所以这里设的标志活不到
            # _update_actor。必须同时存到 self 上，由 _update_actor 重新注入。
            #
            # draft 永远不在 policy 前向里训练（train_draft_only 恒 False，
            # enable_draft_training 恒 False）：v3 的 draft 训练发生在
            # update_actor 之后的独立入口，靠 eagle3_collect_only 采到的特征驱动。
            self._eagle3_serial_flags = {'enable_draft_training': False, 'train_draft_only': False}
            batch.extra_info.update(self._eagle3_serial_flags)
            batch.extra_info["eagle3_collect_only"] = train_draft         # draft训练数据采集总闸
            batch.extra_info["global_steps"] = self.global_steps
            self._eagle3_collect_this_step = train_draft

            logger.debug(
                f"[Serial Training] Step {self.global_steps}: actor step"
                f"{' + draft (collect & train)' if train_draft else ''}"
            )

            # 6. compute old_log_prob  ★ 特征采集就藏在这次前向里
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                batch = self._compute_old_log_prob(batch, metrics=metrics)   

            # 7. [OPTIONAL] compute ref_log_prob
            if self.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    batch = self._compute_ref_log_prob(batch, metrics=metrics)

            # 8. [OPTIONAL] compute critic values
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    batch = self._compute_values(batch, metrics=metrics)

            # 9. compute advantage and return
            with marked_timer("adv", timing_raw, color="brown"):
                batch = self._compute_advantage(batch, metrics=metrics)

            # 10. [OPTIONAL] update critic
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    batch = self._update_critic(batch, metrics=metrics)

            # 11a. draft teacher 快照：必须在 update_actor **之前**。
            #      本步采到的 hidden 是 compute_log_prob 那次前向产的，用的是本步
            #      任何一次 mini-batch 更新**之前**的权重。teacher logits 的算法是
            #      lm_head @ 已存 hidden，所以 lm_head 必须来自同一批权重。
            #      放到 update_actor 之后会把「更新后的头」配到「更新前的身体」上，
            #      这个组合不对应任何真实存在过的模型 —— 而且不报错，只表现为
            #      接受率不涨。SpeCo 同样把 sync 放在更新之前
            #      （speco_ray_trainer.py:1891 vs :1895）。
            if train_draft:
                with marked_timer("draft_teacher_snapshot", timing_raw, color="purple"):
                    self.actor_rollout_wg.snapshot_draft_teacher(batch)

            # 11. update actor
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch = self._update_actor(batch, metrics=metrics)

            # === 增加 Actor 步数 ===
            self.actor_steps += 1
            logger.debug(f"[Serial Training] Global Step {self.global_steps}: "
                        f"Actor step {self.actor_steps}/{self.actor_training_steps}")
            print("*"*100)
            print("*"*100)
            print(f"actor 的 {self.actor_steps} 训练完成")
            print("*"*100)
            print("*"*100)

            # 12. draft 训练：必须在 update_actor **之后**。
            #     此时 policy 的激活与梯度已释放，draft 训练的显存与 policy 峰值错开
            #     （这正是 v1/v2 用「独立步」换来的性质，v3 不必再付那次 rollout）。
            #     teacher 快照不在这里取 —— 见 11a，它必须早于 update_actor。
            if train_draft:
                with marked_timer("update_draft", timing_raw, color="purple"):
                    self._update_draft_deferred(batch, metrics=metrics)
                self.draft_steps += 1
                logger.debug(f"[Serial Training] Global Step {self.global_steps}: "
                            f"Draft step {self.draft_steps}/{self.draft_training_steps}")
                print("*"*100)
                print("*"*100)
                print(f"draft 的 {self.draft_steps} 训练完成")
                print("*"*100)
                print("*"*100)
            else:
                # k>1 时 draft 只在 global_steps % k == 0 那些步训练，其余步这些键根本不
                # 存在，指标行就会时有时无（k=5 时只有 step 5/10/15... 有），画曲线会断。
                # 这里为未训练的步补 0，让每一步都有值、曲线连续。
                #
                # 注意读数时的代价：0 不代表"loss 降到 0"，只代表"本步没训"。
                # 于是 draft/draft_loss 的曲线会呈锯齿状（k-1 个 0 夹一个真实值），
                # 跨步求平均也会被 0 拉低到真实值的 1/k 左右。要看真实的 loss 走势，
                # 请只取非 0 点，或改看 draft/draft_updates>0 的那些步。
                self._fill_draft_metrics_when_skipped(metrics)

        return batch

    # draft 指标的键名，必须与 worker 侧 update_draft_deferred 产出的一致
    # （engine_workers.py:1025-1032）。加前缀 "draft/" 后与真实训练步同名，
    # 这样 TensorBoard / jsonl 里是同一条曲线，不会分裂成两条。
    _DRAFT_METRIC_KEYS = (
        "draft_loss",
        "draft_loss_first",
        "draft_loss_last",
        "draft_updates",
        "draft_windows",
        "draft_time_s",
    )

    def _fill_draft_metrics_when_skipped(self, metrics: dict) -> None:
        """未触发 draft 训练的步，把 draft 指标补 0，使每步都有值、曲线连续。

        只填缺失的键：万一将来 draft 指标改由别处写入，这里不会覆盖真实值。
        """
        for key in self._DRAFT_METRIC_KEYS:
            metrics.setdefault(f"draft/{key}", 0.0)

    def _update_draft_deferred(self, batch: KVBatchMeta, metrics: dict) -> None:
        """v3：用本步采集的特征训练 draft（在 update_actor 之后调用）。

        与已停用的 ``_update_draft``（v1/v2 独立 Draft 步入口，见下方 P3-DEAD 注释块）的
        区别：后者会触发一次完整的 policy 前向来产 teacher；本方法不跑任何 policy 前向，
        teacher 由冻结的 lm_head 副本从已采集的 hidden 重建。
        """
        output = self.actor_rollout_wg.update_draft_deferred(batch)
        if output is None:
            logger.warning(
                "[Serial Training] Step %s: 本步应训 draft，但 worker 没有返回指标。"
                "可能是没有样本通过采集计划的长度门（response 长度不足），"
                "或采集根本没有触发。",
                self.global_steps,
            )
            # 这一步本该训练却没拿到指标，同样补 0，否则"应训却失败"的步会在曲线上
            # 留下空洞，和"按 k 跳过"的步混在一起分不清。draft_updates=0 是判据：
            # 它为 0 说明本步没有任何 optimizer step，无论原因是跳过还是失败。
            self._fill_draft_metrics_when_skipped(metrics)
            return

        from verl.utils.metric import reduce_metrics

        # 与 _update_actor:2024-2027 一致：rename 之后必须 reduce_metrics，
        # 否则值仍是 list，aggregate_logger.py:30 的 isinstance(v, numbers.Number)
        # 会把它静默丢弃，TensorBoard 的 add_scalar 则会抛异常。
        metrics.update(reduce_metrics(rename_dict(output["metrics"], "draft/")))

    # [P3-DEAD v1/v2 20260829] v1/v2 独立 Draft 步的驱动入口，v3 走 _update_draft_deferred，无调用点。
    # 整体验证通过后删除。
#     def _update_draft(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
#         """Draft 训练步骤（新增方法）：
#         1. Actor forward（冻结）生成 teacher logits + hidden states
#         2. Draft forward + loss 计算
#         3. Draft backward + 参数更新
#         """
#         output: TensorDict = self.actor_rollout_wg.update_draft(batch)
#
#         # 提取 draft metrics
#         if output is not None:
#             from verl.utils.py_functional import rename_dict
#
#             draft_metrics = rename_dict(output["metrics"], "draft/")
#             metrics.update(draft_metrics)
#
#         return batch

    # ------------------------------ abstract methods ------------------------------

    def on_init_end(self):
        """Called after the initialization ends."""
        return

    def on_train_begin(self):
        """Called before the training loop starts."""
        return

    def on_train_end(self):
        """Called after the training loop ends."""
        return

    def on_validate_begin(self):
        """Called before the validation loop starts."""
        return

    def on_validate_end(self):
        """Called after the validation loop ends."""
        return

    def on_step_begin(self):
        """Called at the beginning of each training step."""
        return

    @abstractmethod
    def on_step_end(self):
        """Called at the end of each training step."""
        return

    def _consume_sync_metrics(self) -> dict:
        """Weight-sync stats stashed by ``on_step_end`` (e.g. the delta engines'
        changed ratio / wire payload), merged into this step's logged metrics."""
        metrics = getattr(self, "_pending_sync_metrics", None) or {}
        self._pending_sync_metrics = {}
        return metrics

    def on_sample_begin(self):
        """Called at the beginning of sampling batch from replay buffer."""
        return

    @abstractmethod
    def on_sample_end(self):
        """Called after sampling a batch from replay buffer."""
        return

    # ------------------------------ common methods ------------------------------

    def _get_n_gpus_for_throughput(self) -> int:
        """Return the total number of GPUs used for throughput normalization.

        By default this is the trainer-side GPU count from the resource pool
        manager.  Modes that use additional dedicated GPUs (e.g. separate-async
        standalone rollout) should override this to include them.
        """
        return self.resource_pool_manager.get_n_gpus()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        # Exact refill counts require single-prompt dataloader fetches.
        filter_groups = self.config.algorithm.get("filter_groups", None)
        dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        sync_refill_failed_groups = bool(self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
        requires_exact_refill = self.trainer_mode != "sync" or dapo_enabled or sync_refill_failed_groups
        if requires_exact_refill:
            user_gen_batch_size = self.config.data.get("gen_batch_size", None)
            if user_gen_batch_size not in (None, 1):
                logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
            elif user_gen_batch_size is None:
                logger.info("data.gen_batch_size defaulted to 1.")
            with open_dict(self.config):
                self.config.data.gen_batch_size = 1

        # use gen_batch_size as the batch size for the dataloader if set, otherwise use train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        self.steps_per_epoch = len(self.train_dataset) // self.config.data.train_batch_size

        # adjust total_training_steps
        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        # The LR scheduler steps once per local update, and each global step performs
        # ``parameter_sync_step`` local updates (see ``PPOTrainer.step``). The optimizer's
        # schedule horizon must therefore count optimizer updates.
        optim_total_training_steps = total_training_steps * self.parameter_sync_step
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = optim_total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = optim_total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _init_resource_pool_mgr(self):
        config = self.config
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

        # Add actor rollout worker to mapping
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

        # Add critic worker to mapping.
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # 5. restore TransferQueue state (async modes). Re-issuing the restored in-flight prompts is
        # deferred to fit() to use the agent_loop_manager.
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
            if os.path.exists(tq_ckpt_path):
                logger.info(f"Loading TransferQueue state from {tq_ckpt_path}")
                tq.load_checkpoint(tq_ckpt_path)

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Restart checkpointed pending/running prompt groups from their persisted prompt data."""
        if self.trainer_mode == "sync" or not _tq_supports_checkpoint():
            return 0
        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0

        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        inflight_uid_set = set(inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in inflight_uid_set
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)

        # Treat this as a new dispatch attempt for the resumed training step.
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in inflight_uids]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

        logger.info(
            f"Re-issued {len(inflight_uids)} in-flight prompts for step {self.global_steps}; "
            f"cleared {len(old_trajectory_keys)} old trajectories from partition {partition_id}"
        )
        return len(inflight_uids)

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # save TransferQueue state for async modes so in-flight prompts (already fetched from the
        # dataloader but not yet trained into this checkpoint's weights) survive a restart:
        # finished trajectories are restored as-is, pending/running prompts are re-issued on resume.
        # Requires a TransferQueue release with checkpoint support (see _tq_supports_checkpoint).
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq.save_checkpoint(
                os.path.join(local_global_step_folder, "transfer_queue"),
                metadata={"global_steps": self.global_steps},
            )

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
            # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=len(batch)
            )

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = ["uid", "prompts", "responses", "rm_scores", "reward_model"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
                else:
                    sort_keys.append((key, 0, 0))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            # drive the rollout (vLLM) engine's discrete profiler; NPU/torch rollout trace
            # is written to VLLM_TORCH_PROFILER_DIR only while start/stop bracket generate()
            self.llm_server_manager.start_profile()

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            self.llm_server_manager.stop_profile()

    def _fetch_one_gen_batch(self) -> TensorDict:
        """Fetch one ``gen_batch_size`` chunk from the dataloader."""
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None) -> TensorDict:
        """Fetch and coalesce the requested number of prompts."""
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or train_batch_size
        if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
                f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
            )

        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // gen_batch_size)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        """Register prompts in TransferQueue and dispatch them for generation."""
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        if self.trainer_mode != "sync":
            tq.kv_batch_put(
                keys=list(batch["uid"]),
                partition_id="train",
                tags=tags,
                # Persist prompt data for async checkpoint recovery.
                # TODO: maybe let workers do it?
                fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
            )
        else:
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        """Add an exact number of prompts to the AgentLoopManager."""
        batch = self._next_train_batch(num_prompts)
        return self._submit_batch_to_rollout(batch)

    @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
    def _add_batch_to_generate(self):
        """Add one training batch to the AgentLoopManager."""
        batch = self._next_train_batch()
        self._submit_batch_to_rollout(batch)

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict | None = None) -> KVBatchMeta:
        """Compute the reward score with a colocated reward model."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"

        # 1. read the fields required by the reward model from TransferQueue.
        fields = ["prompts", "responses", "raw_prompt"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        prompts = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

        # 2. rebuild the attention mask aligned with the [prompts | responses] layout.
        prompt_mask = self._lengths_to_mask(prompt_lengths, prompts.size(1))
        response_mask = self._lengths_to_mask(response_lengths, responses.size(1))
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        # `raw_prompt` is a non-tensor field; depending on the TransferQueue backend it
        # comes back as a tensordict LinkedList (a `list` subclass), a NonTensorStack or a
        # numpy array. `list(...)` normalizes all of them to a plain list where each element
        # is one sample's chat-message list (whereas `.tolist()` only exists on numpy/tensors).
        raw_prompts = list(data["raw_prompt"])
        raw_prompt_arr = np.empty(len(raw_prompts), dtype=object)
        raw_prompt_arr[:] = raw_prompts

        rm_input = DataProto(
            batch=TensorDict(
                {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
                batch_size=len(batch),
            ),
            non_tensor_batch={"raw_prompt": raw_prompt_arr},
        )

        # 3. run the reward model (wakes/sleeps the reward model internally).
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        # 4. write rm_scores (and reward extra info) back to TransferQueue.
        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(write_back),
        )

        return batch

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Build a right-padded mask of shape (len(lengths), width) from per-row valid lengths."""
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).to(torch.int64)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences
        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return batch

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)    # RPC ─► 跳到/verl/verl/workers/engine_workers.py:compute_log_prob()
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        batch.extra_info.update(
            {
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the actor network."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "distillation_only": distillation_only,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        # === EAGLE3 串行训练：重新注入标志 ===
        # extra_info 在 _compute_old_log_prob → tq.kv_batch_put 处被整体重建，
        # 串行标志已丢失。必须在调用 worker 前补回，否则 forward_step 读到默认值
        # enable_draft_training=True，Actor 步仍会跑 draft 前向导致 OOM。
        if hasattr(self, '_eagle3_serial_flags'):
            batch.extra_info.update(self._eagle3_serial_flags)

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)

        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
        ]
        moe_lb_metrics_interval = self.config.actor_rollout_ref.rollout.get("moe_load_balance_metrics_interval", 0)
        data = get_metric_data_with_optional_routed_experts(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=fields,
            moe_lb_metrics_interval=moe_lb_metrics_interval,
            global_steps=global_steps,
            accumulator=self._rollout_moe_lb_metrics_accumulator,
            kv_batch_get=tq.kv_batch_get,
        )

        num_turns = np.array(data.pop("num_turns").tolist())
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()
        min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]
        max_global_steps = np.array([tag["max_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]

        # Only fetch speculative decoding stats when rollout writes them. Both MTP and
        # EAGLE3 rollout emit the same per-request spec_* fields (vLLM SpecDecodeStats /
        # sglang meta_info); either speculative path enables the accept-rate/length metrics.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        eagle3_config = getattr(self.config.actor_rollout_ref.model, "eagle3", None)
        spec_rollout_on = (
            (mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout)
            or (eagle3_config is not None and getattr(eagle3_config, "enable_rollout", False))
        )
        if spec_rollout_on:
            spec_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = spec_data.pop("extra_fields").tolist()
            # The rollout omits the spec_* stats when the backend does not report
            # per-request spec-decode stats; leave all three as None in that case.
            if extra_fields and all(
                isinstance(extra_field, dict) and "spec_num_draft_tokens" in extra_field for extra_field in extra_fields
            ):
                spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
                spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
                spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]
                logger.warning("-" * 50)
                logger.warning("DRAFT-ROLLOUT: spec stats extracted, num_samples=%d, first_draft=%s",
                               len(spec_drafts), spec_drafts[0] if spec_drafts else None)
                logger.warning("-" * 50)

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(
            compute_moe_lb_metrics(
                metrics_batch=metrics_batch,
                moe_lb_metrics_interval=moe_lb_metrics_interval,
                global_steps=global_steps,
                accumulator=self._rollout_moe_lb_metrics_accumulator,
            )
        )
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self._get_n_gpus_for_throughput()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # === 串行训练专用 metrics ===
        if self._is_serial_training_enabled():
            metrics.update({
                "training/global_steps": self.global_steps,
                # v3 语义：actor_steps 每步递增（每步都是 Actor 步），
                # draft_steps 每 k 步递增。分母 draft_training_steps = actor // k
                # 仍然对得上，只是它现在表示「draft 训练次数」而非「Draft 步数」。
                "training/actor_steps": self.actor_steps,
                "training/draft_steps": self.draft_steps,
                "training/actor_progress": self.actor_steps / self.actor_training_steps if self.actor_training_steps > 0 else 0,
                "training/draft_progress": self.draft_steps / self.draft_training_steps if self.draft_training_steps > 0 else 0,
            })

        # 4. EAGLE3/MTP speculative-decoding acceptance from the vLLM global StatLogger.
        # vLLM V1 exposes spec-decode counts only in the per-step global
        # scheduler_stats.spec_decoding_stats (never per-request), so we aggregate them
        # via a custom StatLogger living in each rollout server process and surface
        # rollout/acceptance_rate, rollout/mean_acceptance_length, rollout/num_draft_tokens,
        # rollout/num_accepted_tokens. Empty dict when spec decode is off / no drafts.
        try:
            spec_metrics = self.llm_server_manager.collect_spec_decode_metrics()
            if spec_metrics:
                metrics.update(spec_metrics)
        except Exception as e:
            logger.warning(f"[eagle3] Failed to collect spec-decode acceptance metrics: {e}")

        # EAGLE3 policy-verify timing (target forward / forward+rejection ms), accumulated
        # per verify step in each rollout worker (vllm_ascend model_runner) and pulled via
        # collective_rpc. Produces rollout/policy_forward_ms_{mean,total} and
        # rollout/policy_forward_rejection_ms_{mean,total}. Empty when spec decode is off.
        try:
            pv_metrics = self.llm_server_manager.collect_policy_verify_timing()
            if pv_metrics:
                metrics.update(pv_metrics)
        except Exception as e:
            logger.warning(f"[eagle3] Failed to collect policy-verify timing metrics: {e}")

        # 4a. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/metric_utils.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

        # 4b. EAGLE3 draft-side metrics grouped under one eagle3/ prefix
        # (eagle3/draft_loss mirrors actor/draft_loss; eagle3/spec_accept_length|rate
        # mirror rollout/spec_*). No-op for a pure policy run.
        metrics.update(
            compute_draft_metrics(
                metrics=metrics,
                spec_drafts=spec_drafts,
                spec_accepts=spec_accepts,
                spec_verifies=spec_verifies,
                non_padding_mask=non_padding_mask,
            )
        )

        # 5. off-policy staleness metrics
        #   global_steps is the model weight version (one update_weights per global_step), and
        #   min/max_global_steps are the versions a trajectory was generated across, so all quantities
        #   below are already in model-version units.
        #   - trajectory_spans: how many distinct model versions a single trajectory was
        #     generated across (1 == fully generated on a single version). This captures the
        #     within-trajectory policy inconsistency caused by partial rollout / continuation.
        #   - trajectory_staleness: how many model versions the trajectory lags behind the
        #     *current* policy. A trajectory spans versions [min_global_steps, max_global_steps],
        #     so the lag is a range: the freshest weights used give the lower bound
        #     (global_steps - max_global_steps) and the oldest weights the worst case
        #     (global_steps - min_global_steps). We log the lower bound as the primary metric.
        trajectory_spans = max_global_steps - min_global_steps + 1
        trajectory_staleness = (global_steps - 1) - max_global_steps
        trajectory_staleness_worst = (global_steps - 1) - min_global_steps
        metrics.update(
            {
                "training/off_policy/trajectory_spans/mean": trajectory_spans.mean(),
                "training/off_policy/trajectory_spans/max": trajectory_spans.max(),
                "training/off_policy/trajectory_spans/min": trajectory_spans.min(),
                "training/off_policy/trajectory_staleness/mean": trajectory_staleness.mean(),
                "training/off_policy/trajectory_staleness/max": trajectory_staleness.max(),
                "training/off_policy/trajectory_staleness/min": trajectory_staleness.min(),
                "training/off_policy/trajectory_staleness_worst/mean": trajectory_staleness_worst.mean(),
                "training/off_policy/trajectory_staleness_worst/max": trajectory_staleness_worst.max(),
                "training/off_policy/trajectory_staleness_worst/min": trajectory_staleness_worst.min(),
            }
        )


TRAINER_REGISTRY: dict[str, type[PPOTrainer]] = {}


class SerialTrainingScheduler:
    # """串行训练调度器：决定每个 step 训练 Actor 还是 Draft

    # 【调度策略】每 k 个 Actor step 后，训练 1 个 Draft step
    # - step 1, 2, ..., k: Actor
    # - step k+1: Draft
    # - step k+2, k+3, ..., 2k+1: Actor
    # - step 2k+2: Draft
    # - ...

    # 示例（k=5）：
    # - step 1,2,3,4,5 → Actor
    # - step 6 → Draft
    # - step 7,8,9,10,11 → Actor
    # - step 12 → Draft
    # """
    def __init__(self, k: int = 5):
        """
        Args:
            k: Actor 训练步数与 Draft 训练步数的比例（k:1）
        """
        self.k = k

    def should_train_actor(self, global_steps: int) -> bool:
        """v3：每一步都训练 Actor。

        v1/v2 让 Actor 步和 Draft 步交替，于是 Draft 步在 trainer_base.py:790
        白跑一次 rollout（实测占该步 89% 时间）。v3 取消交替：draft 改为搭车，
        所以这里恒为 True，保留方法只是为了不破坏既有调用方。
        """
        return True

    def should_train_draft(self, global_steps: int) -> bool:
        """本步是否要采集特征并训练 draft（每 k 步一次）。

        周期是 k 而不是 v1/v2 的 k+1 —— 少掉的那一步正是被取消的独立 Draft 步。
        """
        return self.k > 0 and (global_steps % self.k) == 0


def register_trainer(name: str):
    """Class decorator that registers a :class:`PPOTrainer` subclass under ``name``."""

    def decorator(cls: type[PPOTrainer]) -> type[PPOTrainer]:
        if not (isinstance(cls, type) and issubclass(cls, PPOTrainer)):
            raise TypeError(f"register_trainer expected a PPOTrainer subclass, got {cls!r}")
        existing = TRAINER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Trainer name '{name}' is already registered to {existing.__name__}; "
                f"cannot re-register it to {cls.__name__}."
            )
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer_cls(name: str) -> type[PPOTrainer]:
    """Return the :class:`PPOTrainer` subclass registered under ``name``."""
    try:
        return TRAINER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(TRAINER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown trainer '{name}'. Available trainers: {available}.") from None

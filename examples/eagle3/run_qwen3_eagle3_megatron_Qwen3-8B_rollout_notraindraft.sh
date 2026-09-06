#!/usr/bin/env bash
# EAGLE3 | Qwen3-8B | 串行训练 V3（SpeCo 式搭车采集 + 延后训练）| Megatron | Ascend NPU
# ===========================================================================
# 本脚本是 run_qwen3_eagle3_megatron_Qwen3-8B_serial-training.sh 的 V3 版，
# 对应分支 feature/eagle3-serial-training 的 v3 实现（commit >= 9196101d）。
#
# ---- V3 与 V1/V2 的语义差异（重要，配置口径变了）----
#   V1/V2：Actor 步 / Draft 步交替，周期 k+1；Draft 步独立采一个 rollout batch
#          （该次推理占 Draft 步 89% 时间，纯浪费）。total = actor + draft。
#   V3   ：没有独立 Draft 步。每一步都是完整 Actor 步；每 k 步（global_steps % k == 0）
#          额外做两件事，均不产生额外前向：
#            ① compute_log_prob 那次前向里顺路采集 hidden 窗口（collect_plan/feature_store）
#            ② update_actor 之后用采到的特征训 draft（frozen lm_head 重建 teacher）
#          total_training_steps == actor_training_steps；draft 训练次数 = actor // k。
#
# ---- 本版包含的修复（相对 641a7c93 之前）----
#   P0-1 (a01e5338): 配置校验对齐 v3（不再要求 k+1 整除，k=5/actor=100 可跑）
#   P0-2 (a01e5338): draft DDP 就绪同步（某 rank 采空时全体跳过，防死锁）
#   P1-1 (9196101d): draft 输入 token 相对 hidden 左移一位（EAGLE 对齐，
#          与 vLLM 推理喂法一致；错位配对已实证会把接受率往下拉）
#
# ---- 本次验证要看的指标 ----
#   draft/draft_loss           : 应从 ~2.x（窗口式采集的量纲）随训练缓慢下降
#   rollout/mean_acceptance_length 等 spec 指标 : 至少不应随训练下降（P1-1 修复的验收点）
#   drafter 采集日志           : [DRAFT-COLLECT] stashed N/M window(s)
#   timing_s/update_draft      : 应远小于 v1/v2 的 Draft 步耗时（无额外 rollout/前向）
#
# Usage:
#   bash examples/eagle3/run_qwen3_eagle3_megatron_Qwen3-8B_serial-training-v3.sh
#   # 覆盖参数示例：
#   ACTOR_TRAINING_STEPS=100 ACTOR_STEPS_PER_DRAFT_STEP=5 DRAFT_LR=1e-5 \
#     bash examples/eagle3/run_qwen3_eagle3_megatron_Qwen3-8B_serial-training-v3.sh
# ===========================================================================
set -xeuo pipefail

# ===== 关键环境变量必须最先设置 =====
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}  #,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60500
export HCCL_NPU_SOCKET_PORT_RANGE=61000-62000
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DISABLE_L2_CACHE=1
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export VLLM_USE_V1=1

# ============================================
# Ray 集群预检查和启动
# ============================================
RAY_RUNNING=false
if ray status >/dev/null 2>&1; then
    CURRENT_GPUS=$(python3 -c "import ray; ray.init(address='auto'); g=ray.cluster_resources().get('GPU',0); ray.shutdown(); print(g)" 2>/dev/null || echo "0")
    if [ "$CURRENT_GPUS" = "16.0" ]; then
        echo "OK: Ray cluster already running with 16 GPUs"
        RAY_RUNNING=true
    else
        echo "Ray cluster has $CURRENT_GPUS GPUs (expected 16); restarting..."
        ray stop --force 2>/dev/null || true
        sleep 3
    fi
fi
if [ "$RAY_RUNNING" = false ]; then
    ray start --head --num-gpus=16 --resources='{"NPU": 16}'
    sleep 2
fi

# ===== 路径 =====
POLICY_PATH=${POLICY_PATH:-/home/weight/Qwen3-8B}
DRAFT_PATH=${DRAFT_PATH:-/home/weight/qwen3_8b_eagle3}

# ===== 硬件（Qwen3-8B Dense: 16 卡, TP4 x DP4）=====
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# ===== EAGLE3 draft 训练总开关 =====
DRAFT_ENABLE_TRAIN=${DRAFT_ENABLE_TRAIN:-False}
EAGLE3_ENABLE_ROLLOUT=${EAGLE3_ENABLE_ROLLOUT:-True}   # rollout 侧投机解码开启（要看接受率）
TTT_LENGTH=${TTT_LENGTH:-1}

# ===== V3 串行训练调度 =====
# ENABLE_SERIAL_TRAINING : True = v3 搭车模式；False = 并行模式（draft 每 micro-batch 同步训）
# ACTOR_TRAINING_STEPS   : 总训练步数（v3: total == actor；每步都是完整 Actor 步）
# ACTOR_STEPS_PER_DRAFT_STEP (k) : 每 k 步搭车训一次 draft（周期是 k，不是 v1/v2 的 k+1）。
#   不整除只是最后一个不完整周期少训一次 draft，校验只警告不拦截（P0-1）。
ENABLE_SERIAL_TRAINING=${ENABLE_SERIAL_TRAINING:-False}
ACTOR_TRAINING_STEPS=${ACTOR_TRAINING_STEPS:-100}       # 首轮验证默认 30 步（k=5 -> 6 次 draft 训练）
ACTOR_STEPS_PER_DRAFT_STEP=${ACTOR_STEPS_PER_DRAFT_STEP:-5}

# ===== V3 采集计划参数（当前为代码默认值，记录在此备查）=====
# 位置：verl/models/eagle3/collect_plan.py（对齐 verl-SpeCo speco_base.yaml:58-66）
#   window_train_rows        = 512   每样本训练窗口行数（采 512+1 行，窗口起点 prompt_len-1）
#   window_mode              = front 窗口贴 response 开头
#   sample_rate              = 1.0   不抽样
#   max_samples_per_replica  = 16    每 DP rank 每步最多采 16 条（CollectBudget 跨 micro-batch 记账）
#   max_tokens_per_replica   = 16384 每 DP rank 每步 token 行数上限
# 注意：response 长度 < 513 的样本会被长度门整条丢弃（[DRAFT-COLLECT] 日志可见通过率）。
# 暂无 hydra 面板（engine_workers._eagle3_collect_config 读 actor_config.eagle3_collect，
# ActorConfig 未定义该字段时走上述默认）；要改需改代码默认值。

# ===== Draft 实现与优化器 =====
USE_MEGATRON_DRAFT=${USE_MEGATRON_DRAFT:-True}
DRAFT_TP=${DRAFT_TP:-1}
DRAFT_LR=${DRAFT_LR:-1e-6}
LOSS_WEIGHT=${LOSS_WEIGHT:-1.0}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
ENABLE_VOCAB_COMPRESSION=${ENABLE_VOCAB_COMPRESSION:-True}
DRAFT_VOCAB_SIZE=${DRAFT_VOCAB_SIZE:-32000}
VOCAB_MAPPING_PATH=${VOCAB_MAPPING_PATH:-""}

# ===== V3 draft 训练 batch / 内循环 =====
# DRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU: 前向分块大小（每块处理多少个 513 行窗口）。
# DRAFT_STEPS_PER_TRIGGER (P1-2): 每次 draft 触发做多少次独立 optimizer step，
#   每次从本步采集的窗口池随机抽 DRAFT_TRAIN_BATCH_SIZE_PER_GPU 个窗口。
#   对标 verl-SpeCo speco_base.yaml 的 training.step=10 / batch_size_per_gpu=4。
#   设为 1 可退回"每次触发只更新一次"，用于隔离 P1-1 对齐修复的单变量验证。
# DRAFT_PPO_MINI_BATCH_SIZE: v1/v2 遗留参数，v3 路径不读，不再传。
DRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU=${DRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
DRAFT_STEPS_PER_TRIGGER=${DRAFT_STEPS_PER_TRIGGER:-10}
DRAFT_TRAIN_BATCH_SIZE_PER_GPU=${DRAFT_TRAIN_BATCH_SIZE_PER_GPU:-4}

# ===== V3 采集配额（决定「池子有多大」，与上面「怎么消费池子」是两回事）=====
# 采集发生在第 7 步 compute_log_prob 的前向里，过滤链是：
#   长度门（response < 513 行整条丢弃）→ hash 抽样（sample_rate=1.0 时不生效）
#   → 轮转分配 + 本参数的两个配额
# 两个配额取先到者：
#   DRAFT_COLLECT_MAX_SAMPLES : 每 rank 每步最多存几个窗口
#   DRAFT_COLLECT_MAX_TOKENS  : 每 rank 每步最多存多少行（窗口数 x 513）
# 代码默认 16 / 16384（collect_plan.py 的 DEFAULT_*，对标 verl-SpeCo）。注意
# 16384/513≈31，所以默认下真正卡住采集的是 MAX_SAMPLES 而不是 MAX_TOKENS；
# 调大 MAX_SAMPLES 时必须同步抬 MAX_TOKENS，否则会被后者截住。
# 训练侧用不到池子外的数据：DRAFT_TRAIN_BATCH_SIZE_PER_GPU 设得比池子还大时，
# 代码会退化成"每轮用全部窗口"，多设的部分没有意义。
DRAFT_COLLECT_MAX_SAMPLES=${DRAFT_COLLECT_MAX_SAMPLES:-16}
DRAFT_COLLECT_MAX_TOKENS=${DRAFT_COLLECT_MAX_TOKENS:-16384}

# 采集窗口本身的形状（同为 collect_plan.py 的代码默认）：
# DRAFT_COLLECT_WINDOW_ROWS : 每样本训练窗口的行数。实际存 N+1 行 —— 多出的一行
#   给最后一个训练位置提供 next-token 目标。**它同时就是长度门**：response 短于
#   N+1 行的样本被整条丢弃（不截断也不补齐），所以调小它能放宽长度门、让更多短
#   样本入池；调大则相反，会筛掉更多样本。
# DRAFT_COLLECT_WINDOW_MODE : front = 窗口贴 response 开头（起点 prompt_len-1，
#   那个位置的 hidden 预测第一个 response token）；random = 按 hash 取确定性偏移，
#   同一 (step, sample) 每次结果相同。长回答下 front 只会反复学开头那 512 行，
#   random 能覆盖到中后段。
# DRAFT_COLLECT_SAMPLE_RATE : <1.0 时按 hash 丢弃一部分候选（不是 RNG，可复现）。
#   1.0 = 不抽样。配额已经在限流，一般不用动这个。
DRAFT_COLLECT_WINDOW_ROWS=${DRAFT_COLLECT_WINDOW_ROWS:-512}
DRAFT_COLLECT_WINDOW_MODE=${DRAFT_COLLECT_WINDOW_MODE:-front}
DRAFT_COLLECT_SAMPLE_RATE=${DRAFT_COLLECT_SAMPLE_RATE:-1.0}

# ===== V3 draft LR scheduler（字段在 model.eagle3 下，与 draft_optim_* 同处）=====
# 注意 warmup 按 **draft optimizer step** 计，不是 actor step：
#   总 draft step = (ACTOR_TRAINING_STEPS / k) x DRAFT_STEPS_PER_TRIGGER
#   本配置 30/5 x 10 = 60 步 —— 用 SpeCo 的 warmup=200 会让 LR 全程爬不到头，
#   故默认 constant。要试 cosine 请同时把 warmup 调到总步数的 10% 量级。
DRAFT_LR_SCHEDULER_TYPE=${DRAFT_LR_SCHEDULER_TYPE:-constant}
DRAFT_LR_WARMUP_STEPS=${DRAFT_LR_WARMUP_STEPS:-0}
DRAFT_LR_DECAY_STEPS=${DRAFT_LR_DECAY_STEPS:-60}
DRAFT_LR_MIN_RATIO=${DRAFT_LR_MIN_RATIO:-0.0}

# ===== 数据 / batch / 调度 =====
TRAIN_FILE=${TRAIN_FILE:-/home/t00972278/dataset/dapo-math-17k/dapo-math-17k.parquet}
VAL_FILE=${VAL_FILE:-/home/t00972278/dataset/dapo-math-17k/test.parquet}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}

train_prompt_bsz=${TRAIN_BATCH_SIZE:-32}            # rollout 每步的 prompt 条数
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-8}     # actor 一次 update 的 prompt 条数（内部 x ROLLOUT_N）
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-8}

n_resp_per_prompt=${ROLLOUT_N:-8}
actor_lr=${ACTOR_LR:-1e-6}
total_epochs=${TOTAL_EPOCHS:-1}
# v3: total == actor（_initialize_serial_training_config 会以 actor_training_steps 为准）
total_training_steps=${TOTAL_TRAINING_STEPS:-${ACTOR_TRAINING_STEPS}}
save_freq=${SAVE_FREQ:--1}      # 首轮验证不落 ckpt；正式跑改回 10
test_freq=${TEST_FREQ:--1}
ppo_max_token_len=$((max_prompt_length + max_response_length))

# ===== 并行度 =====
train_tp=${TRAIN_TP:-4}
train_pp=1                      # EAGLE3 门禁：PP 必须为 1
train_cp=1                      # EAGLE3 门禁：CP 必须为 1
train_sp=${TRAIN_SP:-True}
train_ep=1
train_etp=1
gen_tp=${GEN_TP:-1}

project_name=${PROJECT_NAME:-verl_eagle3}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_eagle3_serial_v3}
CKPTS_DIR=${CKPTS_DIR:-"/home/t00972278/verl/ckpts/${project_name}/${experiment_name}"}

# ===== profiling：首轮验证默认关（看曲线不看 trace）；要开设 PROFILE_STEPS="[2,3]" =====
PROFILE_STEPS=${PROFILE_STEPS:-[2]}
PROFILE_ROLLOUT=${PROFILE_ROLLOUT:-True}
PROFILE_ACTOR=${PROFILE_ACTOR:-False}
PROFILE_SAVE_PATH=${PROFILE_SAVE_PATH:-/home/t00972278/desk/eagle3_train/eagle3_result/profile_rollout_notrain/with-stack/gen-tp1-only}

# ===== 指标持久化 =====
export VERL_FILE_LOGGER_ROOT="/home/t00972278/desk/eagle3_train/eagle3_result/logs/metrics"
export TENSORBOARD_DIR="/home/t00972278/desk/eagle3_train/eagle3_result/logs/tb/${experiment_name}"
TRAINER_LOGGER='["console","file","tensorboard"]'

# ===== 参数配置记录：把本次 run 的完整配置落盘，便于对照实验归档 =====
RUN_CONFIG_DIR="/home/t00972278/desk/eagle3_train/eagle3_result/run_configs"
mkdir -p "${RUN_CONFIG_DIR}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_CONFIG_FILE="${RUN_CONFIG_DIR}/${experiment_name}_${RUN_TAG}.txt"
{
    echo "==== EAGLE3 serial-v3 run config | ${RUN_TAG} ===="
    echo "git_branch  : $(git -C /home/t00972278/verl branch --show-current)"
    echo "git_commit  : $(git -C /home/t00972278/verl rev-parse --short HEAD)"
    echo "policy      : ${POLICY_PATH}"
    echo "draft       : ${DRAFT_PATH}"
    echo "serial      : enable=${ENABLE_SERIAL_TRAINING} actor_steps=${ACTOR_TRAINING_STEPS} k=${ACTOR_STEPS_PER_DRAFT_STEP}"
    echo "collect     : max_samples/rank=${DRAFT_COLLECT_MAX_SAMPLES} max_tokens/rank=${DRAFT_COLLECT_MAX_TOKENS} window=${DRAFT_COLLECT_WINDOW_ROWS}(+1) mode=${DRAFT_COLLECT_WINDOW_MODE} rate=${DRAFT_COLLECT_SAMPLE_RATE} 长度门=$((DRAFT_COLLECT_WINDOW_ROWS + 1))"
    echo "draft       : megatron=${USE_MEGATRON_DRAFT} tp=${DRAFT_TP} lr=${DRAFT_LR} deferred_chunk=${DRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU} ttt=${TTT_LENGTH}"
    echo "draft_inner : steps_per_trigger=${DRAFT_STEPS_PER_TRIGGER} batch_per_gpu=${DRAFT_TRAIN_BATCH_SIZE_PER_GPU} (总 draft step = ${ACTOR_TRAINING_STEPS}/${ACTOR_STEPS_PER_DRAFT_STEP} x ${DRAFT_STEPS_PER_TRIGGER})"
    echo "draft_sched : type=${DRAFT_LR_SCHEDULER_TYPE} warmup=${DRAFT_LR_WARMUP_STEPS} decay=${DRAFT_LR_DECAY_STEPS} min_ratio=${DRAFT_LR_MIN_RATIO}"
    echo "vocab       : compression=${ENABLE_VOCAB_COMPRESSION} draft_vocab=${DRAFT_VOCAB_SIZE}"
    echo "rollout     : spec_decode=${EAGLE3_ENABLE_ROLLOUT} num_spec_tokens=${NUM_SPEC_TOKENS} n=${n_resp_per_prompt} gen_tp=${gen_tp}"
    echo "batch       : train_bsz=${train_prompt_bsz} mini=${train_prompt_mini_bsz} actor_micro=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
    echo "seq         : prompt<=${max_prompt_length} response<=${max_response_length}"
    echo "parallel    : tp=${train_tp} pp=${train_pp} cp=${train_cp} sp=${train_sp} (16 cards -> dp=$((16 / train_tp)))"
    echo "actor_lr    : ${actor_lr}"
    echo "steps       : total=${total_training_steps} save_freq=${save_freq} epochs=${total_epochs}"
    echo "data        : ${TRAIN_FILE}"
    echo "metrics     : ${VERL_FILE_LOGGER_ROOT}/${project_name}/${experiment_name}.jsonl"
    echo "tensorboard : ${TENSORBOARD_DIR}"
} | tee "${RUN_CONFIG_FILE}"

########################### parameter arrays ###########################

DATA=(
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="${POLICY_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.eagle3.enable_train=${DRAFT_ENABLE_TRAIN}
    actor_rollout_ref.model.eagle3.enable_rollout=${EAGLE3_ENABLE_ROLLOUT}
    actor_rollout_ref.model.eagle3.draft_model_path="${DRAFT_PATH}"
    actor_rollout_ref.model.eagle3.loss_weight=${LOSS_WEIGHT}
    actor_rollout_ref.model.eagle3.capture_layer_ids=[]
    actor_rollout_ref.model.eagle3.method=eagle3
    actor_rollout_ref.model.eagle3.num_speculative_tokens=${NUM_SPEC_TOKENS}
    actor_rollout_ref.model.eagle3.draft_optim_lr=${DRAFT_LR}
    actor_rollout_ref.model.eagle3.draft_optim_weight_decay=0.01
    actor_rollout_ref.model.eagle3.draft_optim_clip_grad=1.0
    actor_rollout_ref.model.eagle3.draft_optim_offload=${DRAFT_OPTIM_OFFLOAD:-False}
    actor_rollout_ref.model.eagle3.draft_forward_checkpoint=${DRAFT_FWD_CKPT:-True}
    actor_rollout_ref.model.eagle3.draft_lr_scheduler_type=${DRAFT_LR_SCHEDULER_TYPE}
    actor_rollout_ref.model.eagle3.draft_lr_warmup_steps=${DRAFT_LR_WARMUP_STEPS}
    actor_rollout_ref.model.eagle3.draft_lr_decay_steps=${DRAFT_LR_DECAY_STEPS}
    actor_rollout_ref.model.eagle3.draft_lr_min_ratio=${DRAFT_LR_MIN_RATIO}
    actor_rollout_ref.model.eagle3.enable_vocab_compression=${ENABLE_VOCAB_COMPRESSION}
    actor_rollout_ref.model.eagle3.draft_vocab_size=${DRAFT_VOCAB_SIZE}
    actor_rollout_ref.model.eagle3.vocab_mapping_path="${VOCAB_MAPPING_PATH}"
    actor_rollout_ref.model.eagle3.backend=megatron
    actor_rollout_ref.model.eagle3.ttt_length=${TTT_LENGTH}
    actor_rollout_ref.model.eagle3.draft_pipeline_stage=last
    actor_rollout_ref.model.eagle3.allow_pp_gt_1=False
    actor_rollout_ref.model.eagle3.use_megatron_draft=${USE_MEGATRON_DRAFT}
    actor_rollout_ref.model.eagle3.draft_tensor_parallel_size=${DRAFT_TP}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    # ===== EAGLE3 串行训练 V3 =====
    +algorithm.eagle3.enable_serial_training=${ENABLE_SERIAL_TRAINING}
    +algorithm.eagle3.actor_steps_per_draft_step=${ACTOR_STEPS_PER_DRAFT_STEP}
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}
    # v3: deferred draft 训练的窗口分块大小（见上方参数说明）
    # 用 ++ 而非 +：这些 key 不在 yaml 里（只在 ActorConfig dataclass 中），
    # 单 + 是"新增"，命令行再传一次同名 key 会报 "An item is already at ..."。
    # ++ 是"新增或覆盖"，脚本设默认值、命令行仍可覆盖，两者不打架。
    ++actor_rollout_ref.actor.draft_ppo_micro_batch_size_per_gpu=${DRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU}
    # v3 P1-2: 每次触发的 optimizer step 数 / 每步抽样窗口数
    ++actor_rollout_ref.actor.draft_steps_per_trigger=${DRAFT_STEPS_PER_TRIGGER}
    ++actor_rollout_ref.actor.draft_train_batch_size_per_gpu=${DRAFT_TRAIN_BATCH_SIZE_PER_GPU}
    # v3 采集配额（见上方说明；两者取先到者）
    ++actor_rollout_ref.actor.draft_collect_max_samples_per_replica=${DRAFT_COLLECT_MAX_SAMPLES}
    ++actor_rollout_ref.actor.draft_collect_max_tokens_per_replica=${DRAFT_COLLECT_MAX_TOKENS}
    # v3 采集窗口形状（window_rows 同时是长度门）
    ++actor_rollout_ref.actor.draft_collect_window_train_rows=${DRAFT_COLLECT_WINDOW_ROWS}
    ++actor_rollout_ref.actor.draft_collect_window_mode=${DRAFT_COLLECT_WINDOW_MODE}
    ++actor_rollout_ref.actor.draft_collect_sample_rate=${DRAFT_COLLECT_SAMPLE_RATE}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.sequence_parallel=${train_sp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.profiler.enable=${PROFILE_ACTOR}
    actor_rollout_ref.actor.profiler.all_ranks=False
    actor_rollout_ref.actor.profiler.ranks="[0]"
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True
    actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']"
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True
)

REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.sequence_parallel=${train_sp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    # ---- 直通 vLLM 引擎的两个参数（verl 的 rollout 配置里没有对应字段，
    #      只能走 engine_kwargs.vllm 透传；已用 --cfg job --resolve 验证过链路）----
    # draft 侧采样方式：显式写出来只是把 vLLM 的字段默认值固化下来。
    #   /workspace/vllm/vllm/config/speculative.py:255 本就是 greedy，而
    #   build_eagle3_speculative_config（vllm_rollout/utils.py:853）从不传这个 key，
    #   所以此前运行时的实际值已经是 greedy —— **不要期待任何性能变化**。
    #   写进来是为了：① review 时不必再去 vLLM 源码找默认值；② 将来 vLLM 若改了
    #   默认值（如升级后变 gumbel），我们的行为不会跟着漂。
    #   verl-SpeCo 在 vllm_runtime.py:553 同样硬编码 greedy，此项不是两边的差异来源。
    #   唯一消费点：vllm/v1/worker/gpu/spec_decode/eagle/speculator.py:109
    # +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.draft_sample_method=${DRAFT_SAMPLE_METHOD:-greedy}
    # 异步调度：vLLM SchedulerConfig 的字段，默认 None（即不启用）。开启后调度与
    #   模型前向重叠，可掩盖部分 host 侧调度开销。verl 全仓未涉及此参数，故走透传。
    #   默认这里给 True —— 若怀疑它引入不稳定，export ASYNC_SCHEDULING=False 关掉。
    # +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=${ASYNC_SCHEDULING:-True}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.profiler.enable=${PROFILE_ROLLOUT}
    actor_rollout_ref.rollout.profiler.all_ranks=False
    actor_rollout_ref.rollout.profiler.ranks="[0]"
    actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
    actor_rollout_ref.rollout.profiler.tool_config.npu.contents="['npu','cpu','stack']"
    actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_start=30
    actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_end=60
)

TRAINER=(
    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${experiment_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    trainer.val_before_train=False
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    +trainer.actor_training_steps=${ACTOR_TRAINING_STEPS}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.default_local_dir="${CKPTS_DIR}"
)

PROFILER=(
    global_profiler.tool=npu
    global_profiler.steps="${PROFILE_STEPS}"
    global_profiler.save_path="${PROFILE_SAVE_PATH}"
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ALGORITHM[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${PROFILER[@]}" \
    "$@"

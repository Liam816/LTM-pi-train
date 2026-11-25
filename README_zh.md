# LTM-Pi for RoboMME

[English](README.md) | 简体中文

面向 RoboMME 长时序操作任务的 PI0.5 训练与在线推理实现。上游发布脚本的默认实验使用
RoboMME-10x 10 FPS H.264 video 数据、DreamDojo LAM 400k 双视角 memory，并在训练和
推理时固定隐藏最新 10 个 execution memory token。

仓库包含训练、cache 生成、norm stats 计算和 WebSocket 推理所需源码；DreamDojo
LAM 的最小运行时代码已经放在 `third_party/dreamdojo/`，不需要额外 clone
DreamDojo 或 CD-LAM 仓库。数据和模型权重不随源码发布。

## 论文表 1 的原版 RoboMME（MUSA）

[八卡 Python 配置](configs/train/robomme_1x_musa_8gpu.py) 使用原版 16 个任务、每任务 100 条轨迹和 DreamDojo 双视角记忆。全局 batch 为 128，优化器更新 40,000 次，学习率在 1,000 步 warm-up 后从 2.5e-5 衰减至 2.5e-6。训练与推理固定 lag 5，预测 50 步动作，每 10 步重新规划。对应论文表 1 的平均成功率为 58.54%。

`ROBOMME_1X_RESOURCE_ROOT` 下的 `datasets/video`、`artifacts/memory`、`artifacts/anchors` 和 `assets/lerobot/robomme` 应来自同一份原版训练数据。可以用 `scripts/repack_robomme_v3.py SOURCE DESTINATION` 转换原版 LeRobot v3 导出，再用本仓库的 `bin/prepare.sh` 生成两路缓存、锚点及归一化。MUSA 缓存提取使用 `DREAMDOJO_DEVICE=musa:0` 和 `CACHE_GPUS=0,1,2,3,4,5,6,7`。已准备好数据时，在仓库根目录运行以下命令。

```bash
cfg=configs/train/robomme_1x_musa_8gpu.py
bin/train.sh preflight --cfg "$cfg"
bin/train.sh benchmark --cfg "$cfg" --steps 12
```

在机器的 `.env` 中设置 `ROBOMME_DEFAULT_CFG=configs/train/robomme_1x_musa_8gpu.py` 后，`bin/train.sh` 会选择原版配置。正式训练使用 `run --cfg "$cfg"`。评测导出的 MUSA checkpoint 时，服务端从 checkpoint 读取训练 preset 和 lag。表 1 每个任务做 50 次 rollout，报告 3 个评测随机种子的平均成功率。

## 环境

推荐 Linux、Python 3.11、CUDA 12。完整训练配置为 8×H100；cache 生成和推理可用
单张支持 BF16 的 NVIDIA GPU。

```bash
git clone https://github.com/buaa-colalab/LTM-pi.git ltm_pi_robomme
cd ltm_pi_robomme
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bin/setup.sh
```

最后一条命令会创建同级 `../ltm_pi_robomme-resources` 目录，解析其绝对路径，并直接
写入新 `.env` 的 `ROBOMME_RESOURCE_ROOT`。其他路径都会由该资源根目录自动推导。只有
目录结构不同时，才需要修改这个值或单独覆盖对应变量；已有 `.env` 不会被覆盖。也可将
自定义 resource 目录作为 `bin/setup.sh` 的第一个参数传入。

`.env` 只属于当前机器，已明确排除在 Git 之外，不要提交。公开仓库只保留可移植的
`configs/paths.env.example` 模板。发布前运行 `make release-check`，它会拒绝被 Git
跟踪的本机绝对路径、Hugging Face token、代理凭据和环境配置文件。

## 数据与权重

四项外部权重与数据的下载地址如下：

| 资源 | 下载链接 | 默认存放位置 |
| --- | --- | --- |
| PI0.5 base 参数 | [OpenPI model checkpoints](https://github.com/Physical-Intelligence/openpi#model-checkpoints) (`gs://openpi-assets/checkpoints/pi05_base/params`) | `ltm_pi_robomme-resources/models/pi05/params/` |
| DreamDojo LAM 400k 权重 | [LAM_400k.ckpt](https://huggingface.co/nvidia/DreamDojo/resolve/main/LAM_400k.ckpt?download=true) | `ltm_pi_robomme-resources/models/dreamdojo/` |
| RoboMME-10x video 数据集 | [maxenceSUN/RoboMME-10x](https://huggingface.co/datasets/maxenceSUN/RoboMME-10x) | `ltm_pi_robomme-resources/datasets/video/` |
| 预计算 DreamDojo transition cache | [maxenceSUN/RoboMME-10x_Dreamdojo_transition_cache](https://huggingface.co/datasets/maxenceSUN/RoboMME-10x_Dreamdojo_transition_cache) | `ltm_pi_robomme-resources/artifacts/memory/` |

下载后按下面的目录放置：

```text
ltm_pi_robomme-resources/
├── artifacts/
│   └── memory/          # 可选：预计算 DreamDojo transition cache
├── datasets/
│   └── video/            # H.264 RoboMME-10x，用于 cache、norm stats 和训练
└── models/
    ├── pi05/params/
    └── dreamdojo/LAM_400k.ckpt
```

DreamDojo checkpoint 的 SHA-256 必须为：

```text
d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f
```

### 下载预计算 memory cache（推荐）

这份公开 cache 由 RoboMME-10x 16 任务 × 1,000 episodes 数据和上述
DreamDojo checkpoint 生成。`bin/setup.sh` 创建 `.env` 后，可直接下载到
配置好的 cache 目录：

```bash
set -a
source .env
set +a
hf download maxenceSUN/RoboMME-10x_Dreamdojo_transition_cache \
  --repo-type dataset \
  --local-dir "${ROBOMME_MEMORY_CACHE}"
```

该数据集是公开的，无需登录。下载内容包含 `latents.npy`、
`frame_index.npz` 和 `manifest.json`，因此可跳过耗时的 GPU
`bin/prepare.sh memory` 阶段。其中不包含 anchor cache 和 norm stats，
仍需在本地生成并完成校验：

```bash
bin/prepare.sh anchors
bin/prepare.sh norm-stats
bin/validate.sh
```

只有在 RoboMME 数据、transition 规则和 DreamDojo encoder 全部一致时，
才能复用这份 cache；任一输入发生变化时都应重新计算 memory cache。

## 准备 cache 与 norm stats

如果已经下载了上一节的预计算 memory cache，不要再运行
`bin/prepare.sh all` 或 `bin/prepare.sh memory`。只需按上一节执行
`anchors`、`norm-stats` 和 `bin/validate.sh`。本节后续命令用于从零开始
生成 memory cache。

以下命令会依次生成双 anchor、head+wrist DreamDojo memory cache 和训练所需的
state/action norm stats。推荐在 8 卡机器上为 cache 设置 `CACHE_GPUS`；未设置时仍使用
`DREAMDOJO_DEVICE` 指定的单卡（默认 `cuda:0`）：

```bash
CACHE_GPUS=0,1,2,3,4,5,6,7 \
CACHE_PARALLEL_VIEWS=1 \
PAIR_BATCH_SIZE=128 \
PREPROCESS_CHUNK_SIZE=32 \
CACHE_PREFETCH_EPISODES=2 \
bin/prepare.sh all
```

也可以分开运行，便于断点检查：

```bash
bin/prepare.sh anchors
CACHE_GPUS=0,1,2,3,4,5,6,7 \
CACHE_PARALLEL_VIEWS=1 \
PAIR_BATCH_SIZE=128 \
PREPROCESS_CHUNK_SIZE=32 \
CACHE_PREFETCH_EPISODES=2 \
bin/prepare.sh memory
bin/prepare.sh norm-stats
```

`memory` 从 video 数据的 H.264 `image` 和 `wrist_image` 流解码帧，分别提取 32-D
DreamDojo posterior mean，再拼接成 64-D FP32 cache；`norm-stats` 只读取同一数据集的
parquet 状态/动作列，不解码视频。输出会直接写入
`${ROBOMME_ASSETS_ROOT}/${ROBOMME_ASSET_ID}`，无需再手动复制到仓库内的 `assets/`。

生成完成后验证：

```bash
bin/validate.sh
```

cache 的对齐、字段和 manifest 规则见 [docs/data-contract.md](docs/data-contract.md)，
DreamDojo 模型结构与版本固定在 `configs/dreamdojo_lam400k.json`。

## 训练

先运行只读预检，再启动训练：

```bash
bin/train.sh preflight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bin/train.sh launch
bin/train.sh status
```

默认配置：PI0.5、action horizon 50、fixed lag 10、global batch 128、120k steps。
checkpoint 和日志默认写入仓库下的 `checkpoints/`、`runs/`、`artifacts/`，均已被
Git 忽略。W&B 默认离线；需要联网时在 `.env` 加 `WANDB_MODE=online`。

### Python 配置和 S5000 短测

`configs/train/robomme_10x_official.py` 对应原版训练参数。复制配置后可以在
`model`、`data`、`paths`、`train` 和 `checkpoint` 字典中修改实验。
`bin/train.sh show --cfg <配置.py>` 会打印解析后的参数。

S5000 使用 `configs/train/robomme_10x_musa.py`。进入参考项目的 MUSA 环境，
在机器本地的 `.env` 中设置 `MUSA_PYTHON`、`PI05_BASE_PARAMS` 和
`OPENPI_DATA_HOME`。配置里的 `cuda_visible_devices` 同时控制 MUSA 可见设备和
MCCL 进程数。四卡配置每卡每个微批 1 条样本，累积 32 个微批达到有效 batch 128。

```bash
bin/train.sh smoke --cfg configs/train/robomme_10x_musa.py
```

`smoke` 默认只更新两步，每卡每步 1 条样本，在单独的目录保存并回读 checkpoint。
`ROBOMME_SMOKE_RESOURCE_ROOT` 可指向小规模的真实数据、配套 memory、anchor
及 norm stats。常规训练继续读取 `paths` 中的完整数据。MUSA checkpoint 中的
`rank-*.pt` 用于相同卡数下恢复训练，`params/` 和 `assets/` 供现有策略服务读取。

## 推理

先验证训练 checkpoint 与 DreamDojo cache/encoder 是否匹配：

```bash
bin/serve.sh --checkpoint /path/to/checkpoint/10000 --preflight-only
```

启动单卡服务：

```bash
CUDA_VISIBLE_DEVICES=0 bin/serve.sh \
  --checkpoint /path/to/checkpoint/10000 \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8200
```

在线客户端在同一个 WebSocket 连接内直接发送相机帧和状态。首次写入 history：

```python
{
    "add_buffer": True,
    "images": head_frames[:, None],          # [T, 1, H, W, 3] uint8
    "wrist_images": wrist_frames[:, None],  # [T, 1, H, W, 3] uint8
    "exec_start_idx": 120,
}
```

随后发送当前观测进行推理：

```python
{
    "observation/image": head_rgb,          # [H, W, 3] uint8
    "observation/wrist_image": wrist_rgb,  # [H, W, 3] uint8
    "observation/state": state,             # [8] float32
    "prompt": "task instruction",
}
```

服务直接返回 `[50, 8]` action chunk。连接内 history 会持续累积；新连接或 `reset`
会清空 history。协议细节见 [docs/inference-protocol.md](docs/inference-protocol.md)。

## RoboMME 批量评测

批量评测器直接驱动 RoboMME 仿真环境，不读取或生成 NPZ 输入。在精简的 Ubuntu 主机上
先安装 OpenCV 所需的运行库，再为外部 benchmark 建立独立环境：

```bash
sudo apt-get install -y libgl1
git clone https://github.com/RoboMME/robomme_benchmark.git /path/to/robomme_benchmark
cd /path/to/robomme_benchmark
uv sync
uv pip install -e .
```

在本仓库的 `.env` 中加入 benchmark 路径和待评测 checkpoint：

```bash
ROBOMME_BENCHMARK_ROOT=/path/to/robomme_benchmark
ROBOMME_EVAL_CHECKPOINT=/path/to/checkpoint/STEP
```

然后执行预检并启动：

```bash
bin/evaluate.sh preflight
bin/evaluate.sh launch
bin/evaluate.sh status
bin/evaluate.sh attach
```

`preflight` 会检查 release checkpoint、16×50 个 test episode、Python 依赖和 GPU
渲染。`launch` 还会确认所选 GPU 和端口空闲，为每张 GPU 启动一个 policy server，先跑
一条 smoke rollout，通过后才建立完整的 800-job 队列；已有结果目录绝不会被复用。
Vulkan 默认使用系统 loader 自动发现设备，代码中不再内置特定发行版的 ICD 或动态库
路径。只有目标机器确实需要时，才设置 `VK_ICD_FILENAMES` 或
`SAPIEN_VULKAN_LIBRARY_PATH` 覆盖。

默认使用 GPU 0 和 12 个 simulator worker。多卡时可在 `.env` 中设置，例如
`ROBOMME_EVAL_GPUS=0,1,2,3` 和 `ROBOMME_EVAL_WORKERS_PER_GPU=12`。worker 每执行 10
步重新规划；SQLite WAL 队列支持 heartbeat、失联任务回收和最多三次尝试。失败 rollout
保存在 `evaluations/<run>/videos/`，实时及最终指标、成功率和 ETA 写入
`evaluations/<run>/summary.json`。W&B 默认关闭。`bin/evaluate.sh status` 会同时输出
队列摘要、tmux pane 和 GPU 使用率。

## 检查

```bash
make check
bin/validate.sh --code-only
bin/validate.sh --checkpoint /path/to/checkpoint/STEP
```

## 许可与引用

本仓库包含修改后的 OpenPI 代码及 DreamDojo 的最小运行时模块。许可和第三方说明
见 `LICENSE`、`LICENSE_GEMMA.txt`、`third_party/dreamdojo/LICENSE` 和 `NOTICE.md`；
引用信息见 `CITATION.cff`。

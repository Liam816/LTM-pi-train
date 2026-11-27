# LTM-π（DreamDojo）原版 RoboMME 评测手册

**默认实际运行 policy seeds `0 → 42 → 7` 三组。** 用户只说“跑评测”时，按这个顺序逐组运行。每组跑官方 `test` 的 16 个任务，每任务 50 个场景，共 800 回合。三组共 2,400 回合。用户明确指定 seed 时，按指定范围执行。

用 policy seed 派生每个回合的策略随机数。场景和环境 seed 来自固定的官方 `test` 元数据。单组完整跑完后可报告该 seed 的成功率，三组汇总以 2,400 个有效回合为分母。本次已验收的 `seed=42` 为 **55.00%（440/800）**。补齐同一 checkpoint 的三组结果时，可复用这组已验收结果，再跑 seed 0 和 7。这轮结果保存在已验收运行目录的 `full/summary.json` 和 `full/jobs.sqlite3`。

## 节点、权重与评测配置

| 项 | 已验证的设置 |
|---|---|
| 实验节点 | `worker36021`，SSH 堡垒机设备列表选该节点，登录账户 `mccuseradmin` |
| 训练 workdir | `/home/mccuseradmin/heping/projects/LTM-pi-train-wd/LTM-pi-train` |
| branch / commit | `main` / `bcf9616981d4c508028b6420e8d38346bef09258` |
| 训练 cfgfile | `/home/mccuseradmin/heping/projects/LTM-pi-train-wd/LTM-pi-train/configs/train/robomme_1x_musa_8gpu.py` |
| checkpoint | `/data/heping/ltm_pi_robomme_1x/checkpoints/pi05_robomme_1x_dreamdojo_fixedlag5_h50/ltm_pi_robomme_1x_dreamdojo_lag5_musa_8gpu/40000` 的末步 EMA |
| 官方 benchmark | `RoboMME/robomme_benchmark`，提交 `45efbf21036b40af201b24a6d5d37f0d1b088c64`，split 为 `test` |
| 策略与记忆 | front、wrist 双视角，DreamDojo LAM 400K，固定 lag 5，预测 50 步，每次执行前 10 步，去噪 10 步，最多 1,300 个执行步，policy batch 1 |
| 策略运行环境 | 8 张 MTT S5000，物理 GPU 0–7，每卡一路 PyTorch/MUSA BF16。`quicforge` 提供动态库，评测 Python 为 `/data/heping/robomme_musa/venv/bin/python` |
| 仿真环境 | `codex-local/heping-robomme-cpu:20261029`，Docker `runc` CPU 仿真，Mesa lavapipe 渲染。既有环境目录为 `/data/yjwang/heping/official_robomme_20261029` |
| 记忆资源 | `/data/heping/ltm_pi_robomme_1x/artifacts/memory/manifest.json` 和 `/data/heping/ltm_pi_robomme_1x/weights/LAM_400k.ckpt` |
| 已验收运行目录 | `/data/heping/ltm_pi_robomme_1x/evaluations/robomme_40k_seed42_musa8` |

## 已验证的实际链路

`runtime/start_seed42.sh` 激活 `quicforge`，校验训练提交和资源，并使用 `export_musa_ema.py` 将 Orbax EMA 转为 MUSA 策略可读的 `policy-40k-ema.safetensors`。在同一 checkpoint 上复跑时可共用已导出的权重。

`runtime/evaluate_seed42.py` 启动 MUSA 策略和 DreamDojo 编码服务。每卡一个 WebSocket 服务，监听 `127.0.0.1:18420–18427`。`runtime/cpu_eval.py` 在 CPU 容器里调用仓库 `scripts/eval_robomme.py`，直接加载官方场景，按实际仿真结果给每个回合计分。短测用 GPU 0 跑 BinFill episode 0，最多 21 步，检查真实观测、记忆更新和动作回传。短测达到 21 步上限是有效的闭环检查结果。正式阶段用 8 张卡动态领取 800 个测试回合。

策略端和仿真端共用运行目录下的 `simdeps`。这次部署补齐了 `websockets 15.0.1` 和 `msgpack 1.2.4`，并让 MUSA 预处理接收原始 NHWC 图像。下面的命令从已验收目录复制这些运行时文件。原有 `evaluate_seed42.py` 将 seed 写成常量，命令会改每个新目录里的副本。

## 启动评测

在 `worker36021` 的 Bash 执行。先用 `mthreads-gmi --query-gpu=index,memory.used --format=csv,noheader` 确认 0–7 可用，并确认端口 18420–18427 空闲。下方命令创建新的批次目录，在一个 tmux 会话中串行跑三组。每组的正式结果、短测、日志和 SQLite 队列写到各自的 `seed-N` 目录。

```bash
set -euo pipefail
eval_root=/data/heping/ltm_pi_robomme_1x/evaluations
batch=$(mktemp -d "$eval_root/robomme_40k_multiseed.XXXXXXXX")
session="ltm-robomme-$(basename "$batch")"

cat > "$batch/run.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

batch=$(cd "$(dirname "$0")" && pwd)
reference=/data/heping/ltm_pi_robomme_1x/evaluations/robomme_40k_seed42_musa8
checkpoint=/data/heping/ltm_pi_robomme_1x/checkpoints/pi05_robomme_1x_dreamdojo_fixedlag5_h50/ltm_pi_robomme_1x_dreamdojo_lag5_musa_8gpu/40000
revision=45efbf21036b40af201b24a6d5d37f0d1b088c64
reuse=0
if [[ ${1:-} == --reuse-seed42 ]]; then
    reuse=1
    shift
fi
if [[ $# -eq 0 ]]; then
    set -- 0 42 7
fi
for seed in "$@"; do
    case "$seed" in
        0|42|7) ;;
        *) printf 'Unsupported policy seed %s\n' "$seed" >&2; exit 2 ;;
    esac
done

verify_seed() {
    python3 - "$1" "$2" "$checkpoint" <<'PY'
import json
from pathlib import Path
import sys

run, seed, checkpoint = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
status = json.loads((run / "full/status.json").read_text())
summary = json.loads((run / "full/summary.json").read_text())
assert status["phase"] == "complete" and status["scored"] == 800, run
assert status["model_seed"] == seed and status["checkpoint"] == checkpoint, run
assert summary["totals"]["success"] + summary["totals"]["failure"] == 800, run
assert all(summary["totals"][k] == 0 for k in ("error", "pending", "running")), run
assert len(summary["tasks"]) == 16, run
assert all(row["success"] + row["failure"] == 50 for row in summary["tasks"]), run
PY
}

for seed in "$@"; do
    run="$batch/seed-$seed"
    if [[ -e "$run" ]]; then
        verify_seed "$run" "$seed"
        continue
    fi
    if [[ "$seed" == 42 && "$reuse" == 1 ]]; then
        verify_seed "$reference" 42
        ln -s "$reference" "$run"
        continue
    fi

    mkdir -p "$run/runtime"
    cp -a "$reference/runtime/start_seed42.sh" "$reference/runtime/evaluate_seed42.py" \
        "$reference/runtime/serve_musa.py" "$reference/runtime/cpu_eval.py" \
        "$reference/runtime/export_musa_ema.py" "$run/runtime/"
    cp -a "$reference/benchmark-$revision" "$reference/simdeps" "$run/"
    test -f "$reference/policy-40k-ema.safetensors"
    ln -s "$reference/policy-40k-ema.safetensors" "$run/policy-40k-ema.safetensors"
    python3 - "$run/runtime/evaluate_seed42.py" "$seed" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
seed = int(sys.argv[2])
source, changed = re.subn(r"^SEED = 42$", f"SEED = {seed}", path.read_text(), flags=re.M)
assert changed == 1
assert "simdeps" in source
source = source.replace("ltm-robomme-seed42-", f"ltm-robomme-seed{seed}-")
source = source.replace("seed-42 smoke episode", f"seed-{seed} smoke episode")
compile(source, str(path), "exec")
path.write_text(source)
PY
    bash "$run/runtime/start_seed42.sh" > "$run/driver.log" 2>&1
    verify_seed "$run" "$seed"
done

python3 - "$batch" "$@" <<'PY'
from collections import defaultdict
import json
from pathlib import Path
import sys

batch = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
assert len(seeds) == len(set(seeds)), "Duplicate policy seed"
total_successes = 0
task_counts = defaultdict(lambda: {"success": 0, "episodes": 0})
per_seed = {}
for seed in seeds:
    path = batch / f"seed-{seed}" / "full/summary.json"
    result = json.loads(path.read_text())
    wins = result["totals"]["success"]
    total_successes += wins
    per_seed[seed] = {"success": wins, "episodes": 800, "success_rate": wins / 800}
    for task in result["tasks"]:
        row = task_counts[task["task"]]
        row["success"] += task["success"]
        row["episodes"] += task["success"] + task["failure"]
episodes = 800 * len(seeds)
report = {"seeds": seeds, "success": total_successes, "episodes": episodes,
          "success_rate": total_successes / episodes, "per_seed": per_seed, "tasks": dict(task_counts)}
(batch / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Success rate {100 * total_successes / episodes:.2f}% ({total_successes}/{episodes})")
PY
BASH

bash -n "$batch/run.sh"
tmux new-session -d -s "$session" -n controller \
    "bash '$batch/run.sh' >'$batch/driver.log' 2>&1"
printf 'Batch %s\nSession %s\n' "$batch" "$session"
```

如需只跑用户指定的种子，在 tmux 命令中的 `bash '$batch/run.sh'` 后增加 seed，例如 `bash '$batch/run.sh' 42`。补齐既有 40K 结果时，追加 `--reuse-seed42`。该选项在默认三组中验收并复用既有 seed 42。新一轮完整复跑时，直接执行上面的默认命令。

## 进度、验收与故障恢复

- 在批次目录下看 `driver.log` 和 `seed-N/driver.log`。短测状态在 `seed-N/smoke/status.json`，正式状态在 `seed-N/full/status.json`。从 `jobs` 可读到 `pending`、`running`、`success`、`failure` 和 `error`。
- 正式阶段的 `full/summary.json` 提供分任务、分组和总体计数，`full/jobs.sqlite3` 保存逐回合结果。需要调查某一路时查 `full/worker-gpuN.log` 和 `full/server-gpuN.log`。
- 每个 seed 要求 `phase=complete`、800 回合有效、每任务 50 条、运行错误为 0。任务失败和超时均计入分母。默认三组均通过后，批次根目录的 `aggregate.json` 应为 2,400 回合。汇报时标注 seed 列表、成功数、分母和结果路径。
- SSH 断开后可以重连查看原 tmux 会话。若短测或正式阶段退出，先看最近一次 `status.json` 的 `error` 和对应日志。`ModuleNotFoundError: websockets` 时检查 `simdeps/websockets/asyncio/server.py` 以及策略进程的 `PYTHONPATH`。`mthreads-gmi` 查询使用 `--format=csv,noheader`，显存值带 `[MiB]` 单位。
- 确认批次已退出并修复原因后，将故障 seed 的目录改名留档，再以原参数运行该批次的 `run.sh`。三个种子各自保存状态，启动脚本会验收并沿用同一批次里已经完成的组。

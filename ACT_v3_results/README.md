# ACT V3：当前仅启用 Cube Transfer Human

按用户最新要求，先训练和排队评估 `sim_transfer_cube_human`。
任务 B/C（Insertion Scripted/Human）暂停自动下载、预处理、训练和评估，由用户处理数据。
已有下载片段保留于 `downloads/sim_insertion_*/`；这些片段不是完整的 HDF5 数据。
任务开关记录在 `experiment_plan.json` 的各任务 `enabled` 字段，未得到后续明确指示前不自动启用 Insertion。

Cube Human：ACT k=100，seed=0，batch_size=8，lr=1e-5，KL=10，hidden=512，FF=3200，ResNet18，top 相机，从头训练 2000 epochs。
保留官方 ImageNet 预训练骨干，不加载 V1/V2 模型权重。50 条 Human 数据已逐文件完整检查，独立归一化统计和 40/10 轨迹划分保存于 `reports/data/cube_transfer_human_seed0/`。

训练在 A800 CUDA 上，以一个低 CPU 优先级进程与 V2 评估重叠。
**评估同时等待本模型训练成功、V2 真正完成且完整性检查通过。** 不仅检查 finish.txt 存在。
V2 实际已经缩减为三个 seed=0 模型和四组评估，检查器从其真实计划读取数量，不套用旧九模型/十二组。

排队由 `scripts/wait_for_v2_then_run.py` 维护，约每 60 秒只读检查 V2。
它通过排他锁防止重复启动；模型训练与评估监督进程另有 `pipeline.lock`。
队列和监督进程采用独立会话、关闭标准输入并将输出写入日志，终端退出后继续运行。
队列 PID 为 `queue.pid`，监督 PID 为 `pipeline.pid`。

## 监控

在项目根目录运行：

```bash
python ACT_v3_results/scripts/monitor.py
python ACT_v3_results/scripts/monitor.py --watch
cat ACT_v3_results/status.txt
tail -f ACT_v3_results/logs/queue_checks.jsonl
```

实际模型目录以 `reports/run_state.json` 为准。
第一次自动启动因配置键拼写错误在训练前退出，现场保留于 `models/cube_transfer_human_seed0/`。
修复后使用 `models/cube_transfer_human_seed0_attempt1/` 从头训练，完整输出在该目录 `stdout.log`、`stderr.log`，逐 epoch 指标在 `epoch_metrics.jsonl`。
没有从仅含模型权重的检查点冒充恢复完整优化器状态。

## 评估排队与输出

本模型完成后继续等待 V2 验收通过。队列不会运行 Insertion。
V3 使用独立的 OSMesa LLVMpipe 软件渲染库和 MPL 配置目录，导入前设置 `MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`MPLBACKEND=Agg`，不继承 EGL 专用变量。
先在 V3 本任务内验证多相机与只渲染 top 的像素、状态、奖励和随机数一致性。
随后加载本模型的最佳验证检查点与对应 dataset_stats.pkl，开启 temporal aggregation，使用官方 seed=1000，实际运行 50 回合、每回合 400 步。
只有一个有效评估配置时不额外跑并发测速回合。
每组保存真实第一个成功及第一个失败视频；若未出现某类，videos.json 如实记录缺失。

结果位于 `reports/summary.csv`、`reports/episode_results.csv`、`reports/final_report.md`。
Cube 任务通过后写入 `reports/cube_transfer_human_seed0_completion.json`，它不是全局 V3 完成标志。
Insertion 未完成前不创建 `finish.txt`；全局完成仍要求三个任务各训练 2000 轮、三组各评估 50 回合、150 条记录、三张图和全部完整性检查。

V1/V2 模型、日志、统计、视频与完成标志只读，未修改其内容。

# ACT Transfer Cube V2

当前正式协议已按用户要求缩减为 **一个训练种子 seed=0**：三个 ACT 模型（k=1/100/400，各 2000 epochs）和四组正式评估（每组 50×400 步，共 200 回合），与 E1 完全分开。

主实验为 k100_seed0 + temporal aggregation；消融为 k1/k100/k400_seed0，均关闭 temporal aggregation。复用同一个 k100 模型，不重复训练。
原 seed=1/2 的已完成、部分训练和评估文件保留，但全部排除正式统计；其状态在 run_state.json 的 excluded_models / excluded_evaluations 中，变更前计划、报告和代码已保存到 backups/single_seed_scope_*。
保留 seed=0 按最小编号和已有工作延续规则决定，不根据成功率挑选。单训练种子不能估计种子间样本标准差，CSV 相应字段留空，图中不绘制种子间误差棒，也不冒充论文的三种子结果。

**实际协议：按用户最后授权，容器无 graphics 管理权限时使用 CPU 渲染。NVIDIA EGL 未启用。**
训练和推理仍在 A800 CUDA GPU 上；渲染为隔离的 OSMesa LLVMpipe。
因此本轮完成标志将写明 CPU 渲染替代协议，不能解读为满足原 NVIDIA EGL 条件。

当前阶段、三个训练及四个评估状态见 `status.txt` 和 `reports/run_state.json`。
`reports/final_report.md` 在运行中是明确标注的进行中报告，不能当作最终成功结论。
没有完成所有正式任务、200 回合、统计图和完整性校验之前，不会创建 `finish.txt`。

## 监控

在项目根目录执行（只读监控可使用系统 Python）：

```bash
python ACT_v2_results/scripts/monitor.py
python ACT_v2_results/scripts/monitor.py --watch
tail -f ACT_v2_results/logs/pipeline.stderr.log
tail -f ACT_v2_results/logs/resource_monitor.jsonl
```

完整训练/评估输出在对应任务目录 `stdout.log`、`stderr.log`。
训练逐轮记录 `epoch_metrics.jsonl`，原始官方指标和同步计时均保存；当前进度为 `progress.json`。
正式模型目录以 `reports/run_state.json` 为准，重试时可能是 `_attemptN`，不能只按默认目录名读取。
评估逐回合原始奖励为 `episodes.jsonl`，另保存 `episode_XX.npz` 的动作、qpos、奖励及初始方块位置。

调度器通过独立进程会话运行，标准输入关闭、输出重定向，终端退出后仍运行。PID 在 `pipeline.pid`。
监督进程退出/容器被回收属于中断；状态文件不会自动成为完成证据。

## 中断后的继续

先运行 monitor 确认没有旧调度器或它管理的训练进程仍在运行，然后执行：

```bash
source results/activate_act_repro.sh
nohup python -u ACT_v2_results/scripts/pipeline.py --continue_run \
  > ACT_v2_results/logs/continued_pipeline.stdout.log \
  2> ACT_v2_results/logs/continued_pipeline.stderr.log < /dev/null &
```

监督进程通过 `flock` 避免重复运行。已完成任务不重复训练；未完成任务保留原目录，在新 attempt 目录从头训练。
官方检查点只有模型权重，不能恢复完整优化器、RNG 和数据加载器状态；不进行“从中断 epoch 接上”的伪恢复。
上面的继续命令需在旧进程结束后使用；若监控指示还有旧 worker，先核实情况，不能同时启动副本。

## 训练与评估重叠调度

按用户“尽量并行”的要求，调度器采用已完成模型先评估的方式：无需等所有模型全部训练完毕。
单种子缩减后，k100_seed0 与 k1_seed0 的完整训练直接保留，k400_seed0 继续原进程训练；同时启动两个符合条件的 seed=0 评估进程。
`parallel_controls.json` 记录训练并发（最多三）、纯评估阶段并发，以及训练期间允许的评估进程数。
调度器每次循环读取该文件；降低上限只限制新任务启动，不杀死或重启已有任务。
训练/评估参数、数据划分、固定种子和正式评估回合数不变。

历史调度器交接记录见 `reports/parallel_scheduler_migration.json`；交接不终止训练 worker，也不从权重重建优化器。
若需在旧监督进程**已经退出**且相关 worker 仍存活时接管，可使用：

```bash
source results/activate_act_repro.sh
nohup python -u ACT_v2_results/scripts/pipeline.py --continue_run --adopt_running \
  > ACT_v2_results/logs/adopted_pipeline.stdout.log \
  2> ACT_v2_results/logs/adopted_pipeline.stderr.log < /dev/null &
```

接管会校验 worker 的用户、命令、输出目录及进程启动时间，防止 PID 复用。失去父子关系后，OS 退出码无法追回，因此采用 worker 完成标志和产物校验判断完成，明确记录退出码来源。

## 渲染设置

不要只给新 LLVMpipe 目录设置 LD_LIBRARY_PATH 就认为切换成功：Conda Python 的 DT_RPATH 会优先加载旧库。
`scripts/common.py` 的 `worker_environment()` 和 `activate_renderer()` 共同在**本进程**中选择 V2 目录的 OSMesa/GLAPI。
所有训练/评估入口自动在导入官方模块之前完成该设置，不修改 Conda 文件、系统驱动或 CUDA。
不要将这些库目录全局加入 shell 配置。

参考环境：`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`GALLIUM_DRIVER=llvmpipe`、`LP_NUM_THREADS=2`、`MPLBACKEND=Agg`，配合上述库路由。
原激活脚本和 E1 继续保留旧 OSMesa，不受影响。

EGL 阻塞与管理员修复说明仍保留在 `egl_validation/`；本轮复测为 `logs/egl_recheck.log`。
CPU 实测对比见 `reports/software_renderer_comparison.json`、`tests/render_*/validation.json`。
LLVMpipe 软件加速不能标为 NVIDIA EGL GPU 加速。

## 实验与统计

`experiment_plan.json` 固定三个模型和四组评估；模型共享官方种子 1 产生的 40/10 数据划分。
训练种子在官方 train_bc 中另设为 0。保留官方 ImageNet 预训练 ResNet18 骨干，其余 ACT 参数独立初始化，不加载 E1 模型。
训练测速比较 1/2/3 个独立进程的真实总 epoch 吞吐，最多三进程；评估用同一冒烟模型比较单/双进程，每个初步回合不进入正式统计。

每组正式评估从该模型的最佳验证检查点和配套归一化统计出发；使用官方 seed=1000、50 回合、400 步和最高奖励=4 判据。
k100 时间集成与无时间集成结果保存到不同目录。
视频只选择每组真实出现的第一个成功和第一个失败；没有该类时明确记录缺失。
阶段图按官方奖励阈值统计；CSV 另外提供只读接触证据得到的接触、抬起及左臂接触事件率。

汇总脚本 `scripts/collect_results.py` 自动运行；需要只更新进行中 CSV/报告时可执行：

```bash
python ACT_v2_results/scripts/collect_results.py --partial
```

正式统计文件为 `reports/per_seed_results.csv`、`reports/episode_results.csv`、`reports/summary.csv`。
最终三张图为 `figures/main_success.png`、`figures/chunk_ablation.png` 和 `figures/reward_stage_attainment.png`，报告为 `reports/final_report.md`。
只在监督进程确认全部正式任务完成后执行不带 `--partial` 的最终校验，创建明确标注 OSMesa CPU 的 `finish.txt`。

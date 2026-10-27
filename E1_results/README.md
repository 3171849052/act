# E1 / E3 实际执行产物

本目录对应仅程序化示范、1000 轮、种子 0、每模型 30 回合的四模型快速复现。
真实状态以 `status.txt`、`reports/run_state.json` 和进程存活情况为准。
只有全部训练、原始评估、汇总、图表、中文报告与完整性检查成功，才创建 `finish.txt`。

监控（在 ACT 项目根目录）：

```bash
python E1_results/scripts/monitor.py
cat E1_results/status.txt
tail -f "$(cat E1_results/active_pipeline_log.txt)"
tail -f E1_results/models/M1_ACT_k100/train.stdout.log
```

训练/评估 stdout、stderr 均位于各任务的独立目录；PID、命令、起止时间、退出码和耗时保存在 `job.json`。
全局资源采样位于 `logs/resource_monitor.jsonl`。调度器进程号保存在 `pipeline.pid`。
`backups/imitate_episodes.py.original` 为执行前官方源码备份。项目训练、模型、环境源码保持原样。

正式评估包装只将官方评估回合数和视频保存条件参数化，默认值仍为 50；本次明确 30 回合、仅保存前两个视频。
模型构造器二次解析命令行，因此新参数由独立包装器解析，不传给原构造器。
每个评估目录以符号链接指向对应模型的最佳检查点和统计文件，输出彼此隔离。

最终文件（当前任务运行中时可能尚不存在）：

- `reports/summary.csv`
- `reports/episode_level_results.csv`
- `reports/experiment_manifest.csv`
- `figures/E1_algorithm_comparison.png`
- `figures/E3_chunk_size_ablation.png`
- `reports/final_report.md`
- `reports/integrity_check.json`
- `finish.txt`

启动器与调度器均拒绝覆盖旧目录和成功标志；不要在已有进程运行时再次启动。
发生 OOM 或严重资源争用时仅停止当前调度器拥有的子进程，降低并发，在新的重试目录从头训练。
非资源错误会停止流程、保留日志与检查点并写失败报告，不会生成成功标志。

## 用户授权后的评估加速

当前调度器改为 `scripts/run_accelerated_evaluations.py`，最多三个独立评估进程。相机过滤的像素/状态/奖励/RNG 一致性及短测试速度见 `reports/camera_filter_validation.json`；EGL 实际失败证据与平台侧启用条件见 `reports/egl_enablement.md`。

旧 M1 部分评估完整归档到 `evaluations/M1_ACT_k100_before_acceleration_*`，新 M1 从 0 开始完成 30 回合，并核对旧回合指标。M1 前两个视频保留旧原始文件且在新目录引用，须通过 `legacy_prefix_validation.json` 才能通过最终检查。

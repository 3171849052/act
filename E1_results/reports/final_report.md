# E1、E3 仿真实验快速复现报告

完成检查时间：2026-10-26T22:34:39.147698+08:00。
实验目录：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results`。项目目录：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act`。

## 环境与数据
服务器：Linux-5.15.0-78-th-an-x86_64-with-glibc2.17；GPU：NVIDIA A800 80GB PCIe；PyTorch CUDA runtime：11.8。
Python：`/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs/act-repro/bin/python`；版本：3.8.10 (default, Jun  4 2021, 15:09:15) 
[GCC 7.5.0]。
软件版本：`{"torch": "2.0.1+cu118", "torchvision": "0.15.2+cu118", "mujoco": "2.3.7", "dm-control": "1.0.14", "h5py": "3.10.0", "numpy": "1.23.5", "matplotlib": "3.7.5"}`。
Git 提交：`742c753c0d4a5d87076c8f69e5628c79a8cc5488`。初始工作区见 `reports/initial_git_status.txt` 和 `initial_worktree.diff`。
仅使用 `data/sim_transfer_cube_scripted/` 中的 50 条程序化示范；逐文件完整读取数值及图像并记录 SHA256。未使用人工示范数据。
官方数据加载器按 main() 的种子 1 生成固定 40/10 训练/验证划分，模型训练种子为 0。归一化遵循官方实现，使用全部 50 条数据计算统计量；这包含验证轨迹的信息，是官方代码的复现局限。
MuJoCo 使用 `osmesa`。当前为 CPU 软件离屏渲染，评估速度可能较慢。

## 参数与训练
四模型均为 batch_size=8、num_epochs=1000、lr=1e-5、seed=0、resnet18、顶部相机。ACT 均为 kl_weight=10、hidden_dim=512、dim_feedforward=3200、4 层编码器、7 层解码器、8 个注意力头；仅 k 不同。CNNMLP 使用官方单步行为克隆。未启用混合精度或时间集成。
M1 只训练一次，共用于 E1 和 E3。独立两轮短训练及 50 轮测速均未计入正式实验。最多三个独立训练进程，资源争用/OOM 时降低并发并在独立目录从头重试，不假设无损续训。
50 轮 ACT 测速耗时 53.17 秒，含启动平均 1.063 秒/轮。测速原始 GPU、显存、CPU 数据见 `logs/resource_monitor.jsonl`。
测速资源采样汇总：`{"benchmark_seconds": 53.167640924453735, "average_seconds_per_epoch_including_startup": 1.0633528184890748, "sample_count": 10, "sampling_interval_seconds": 5, "gpu_utilization": {"min": 0.0, "max": 95.0, "mean": 41.3}, "memory_used_mib": {"min": 0.0, "max": 4459.0, "mean": 3660.9}, "cpu_load_1min": {"min": 21.12255859375, "max": 25.2666015625, "mean": 22.9375}, "cpu_count": 112}`。5 秒离散采样不能代表瞬时峰值。
正式训练墙钟耗时 1577.41 秒（0.438 小时）；包括重试的训练进程时长总和 3816.02 秒。

| 模型 | k | 训练状态 | 成功训练耗时(秒) | 评估成功数 | 成功率 | 平均回报 |
|---|---:|---|---:|---:|---:|---:|
| M1_ACT_k100 | 100 | SUCCEEDED | 1181.49 | 12/30 | 40.00% | 274.000 |
| M2_CNNMLP |  | SUCCEEDED | 650.52 | 0/30 | 0.00% | 0.000 |
| M3_ACT_k1 | 1 | SUCCEEDED | 1068.25 | 0/30 | 0.00% | 0.000 |
| M4_ACT_k10 | 10 | SUCCEEDED | 915.76 | 0/30 | 0.00% | 42.733 |

## E1 算法对比
ACT(k=100) 成功率 40.00%，CNNMLP 为 0.00%；差值 40.00 个百分点。此结论仅适用于本次单种子快速实验。
图表：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/figures/E1_algorithm_comparison.png`。

## E3 动作块长度消融
ACT k=1、10、100 成功率依次为 0.00%、0.00%、40.00%。同数据划分和其他训练参数下比较，未进行时间集成。
图表：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/figures/E3_chunk_size_ablation.png`。

## 原始评估、视频和汇总表
四组均采用各自最佳验证检查点及对应统计量，每组完整评估 30 回合。保持官方评估种子 1000、初始状态分布、动作执行、奖励及成功判据（最高奖励等于 4）。每组仅保留前两个回合视频，保存条件不改变评估循环或统计。
评估主循环源码由官方备份仅参数化回合数与视频保存条件；各组 `evaluation_source.py` 可审计。用户授权加速后，另外通过运行时包装过滤 angle/vis 两个无用相机，保持顶部相机原分辨率。项目训练、模型和环境源文件保持原样；运行时观察方法的完整前后源码分别保存在各组 observation_source.original.py / observation_source.py。

### 评估加速与验证
经用户明确授权，四组正式评估采用最多 3 个独立进程并行，每个工作进程 CPU Torch 线程数为 1。每组随机种子、400 步动作、30 回合统计均保持原定义。
相机过滤测试覆盖两组固定初始位置、每组 8 步（含 reset 共 18 个观察）：顶部图像逐像素、qpos/qvel、物理状态、奖励和 NumPy RNG 均完全一致。原耗时 35.174 秒，过滤后 11.912 秒，实测 2.953 倍；这是短测试速度，不作为完整评估加速倍数。
旧 M1 慢速评估中断后保留于 evaluations/M1_ACT_k100_before_acceleration_*，未将旧部分回合计入本次正式成功率。新 M1 从回合 0 重新完成完整 30 回合，并逐项核对旧流程已完成回合的回报、最高奖励和成功判定一致。前两个旧视频复用到新评估目录，关联验证见 legacy_prefix_validation.json；其余组前两个视频为顶部相机单视角。
EGL 已实际重测，但当前容器未暴露 NVIDIA graphics 能力和 EGL 图形库，也无可用 Docker/容器管理入口。未安装依赖、未改系统驱动或 CUDA；因此本次仍使用 OSMesa softpipe。后续容器侧启用方案见 reports/egl_enablement.md。
- M1_ACT_k100：原始结果 `/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M1_ACT_k100/result_policy_best.txt`，完整 stdout/stderr 日志位于同目录；奖励阈值达到率：`{'0': 1.0, '1': 0.9666666666666667, '2': 0.8666666666666667, '3': 0.4, '4': 0.4}`。
  - 回合 0（失败，回报 34，最高奖励 2）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M1_ACT_k100/video0.mp4`。
  - 回合 1（失败，回报 151，最高奖励 2）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M1_ACT_k100/video1.mp4`。
  前两个保存回合没有成功案例视频；未额外补录或代用其他回合。
- M2_CNNMLP：原始结果 `/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M2_CNNMLP/result_policy_best.txt`，完整 stdout/stderr 日志位于同目录；奖励阈值达到率：`{'0': 1.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0}`。
  - 回合 0（失败，回报 0，最高奖励 0）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M2_CNNMLP/video0.mp4`。
  - 回合 1（失败，回报 0，最高奖励 0）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M2_CNNMLP/video1.mp4`。
  前两个保存回合没有成功案例视频；未额外补录或代用其他回合。
- M3_ACT_k1：原始结果 `/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M3_ACT_k1/result_policy_best.txt`，完整 stdout/stderr 日志位于同目录；奖励阈值达到率：`{'0': 1.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0}`。
  - 回合 0（失败，回报 0，最高奖励 0）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M3_ACT_k1/video0.mp4`。
  - 回合 1（失败，回报 0，最高奖励 0）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M3_ACT_k1/video1.mp4`。
  前两个保存回合没有成功案例视频；未额外补录或代用其他回合。
- M4_ACT_k10：原始结果 `/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M4_ACT_k10/result_policy_best.txt`，完整 stdout/stderr 日志位于同目录；奖励阈值达到率：`{'0': 1.0, '1': 0.4, '2': 0.3, '3': 0.0, '4': 0.0}`。
  - 回合 0（失败，回报 174，最高奖励 2）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M4_ACT_k10/video0.mp4`。
  - 回合 1（失败，回报 0，最高奖励 0）：`/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/lt/act/E1_results/evaluations/M4_ACT_k10/video1.mp4`。
  前两个保存回合没有成功案例视频；未额外补录或代用其他回合。

汇总表：`reports/summary.csv`；120 回合明细：`reports/episode_level_results.csv`；配置与校验和清单：`reports/experiment_manifest.csv`。

## 与原始设置的差异与局限
本次采用 1000 轮训练、每模型 30 回合评估、训练随机种子 0，属于 E1、E3 快速复现，与论文原始设置存在差异；未将论文成功率填入本次结果。当前仓库 README 的训练示例为 2000 轮，官方评估代码默认为 50 回合。此处以本地官方源码/README 为可核实对照，不声称逐项重建论文完整协议。
不包含人工示范实验、E2、E4 或时间集成实验；不是论文所有实验的完整复现。单随机种子结果不能代表多随机种子统计结论；30 回合样本量有限，成功率有抽样误差。
训练样本使用官方随机起点和每轮遍历方式；1000 轮只表示官方 epoch 循环完成 1000 次，不等于每条轨迹所有 400 个时刻均被训练遍历。
最终检查重读最佳检查点/统计文件、核对 1000 轮日志与完整指标、30 条原始回报和最高奖励、成功率与阈值计数、全部视频帧、两张图及报告。所有必要任务均通过；若存在重试，失败日志和检查点保留于各独立重试目录。

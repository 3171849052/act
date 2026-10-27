# ACT V3 单种子实验报告



全部三个任务完成。

更新时间：2026-10-27T16:59:37.269087+08:00



训练固定 ACT k=100、seed=0、2000 epochs、batch_size=8、lr=1e-5、KL=10、hidden_dim=512、FF=3200、ResNet18、top RGB。保留官方 ImageNet 预训练骨干，其余参数从头初始化，不加载 V1/V2 检查点。

本轮只使用一个训练种子，不估计或虚构种子间标准差。各任务独立数据、检查点及 dataset_stats.pkl；官方 main() 固定划分种子为 1，各任务使用自己的 40/10 轨迹划分与全 50 条轨迹的官方归一化逻辑。

训练使用 A800 CUDA GPU；仿真使用独立 V3 目录内的 OSMesa LLVMpipe CPU 图形库，环境变量在导入前设置，未修改系统或 Conda 依赖。

按用户最新授权，Cube Human 训练与 V2 评估重叠，以单训练进程和较低 CPU 优先级运行。V3 正式评估等待 V2 的实际计划、状态、完成标志、完整性报告与无在跑正式 worker 检查通过；不假定旧的 9 模型/12 评估计划仍有效。V1/V2 文件仅只读访问。

正式评估每任务 50 回合，全部开启官方 Temporal Ensembling、随机种子 1000；Transfer Cube Human / Insertion Scripted / Insertion Human 分别按 constants.py 使用 400/400/500 步，最高奖励等于环境 max_reward 判成功。

顶部相机过滤按实际启用任务在 V3 独立验证，比较两个初始状态的顶部像素、qpos/qvel、完整物理状态、奖励与 NumPy/Python/Torch RNG；渲染身份和误差见 reports/render_selection.json。保留真实第一例成功和第一例失败视频，缺少某类时如实记录，不用视频选择替代完整 50 回合统计。



| 任务 | 状态 | 成功/回合 | 实测成功率 | 平均回报 | 论文参考 |
| --- | --- | ---: | ---: | ---: | ---: |
| sim_transfer_cube_human | COMPLETED | 29/50 | 58.00% | 379.26 | 50% |
| sim_insertion_scripted | COMPLETED | 4/50 | 8.00% | 354.28 | 32% |
| sim_insertion_human | COMPLETED | 14/50 | 28.00% | 500.62 | 20% |



[ACT 论文 Table I](https://arxiv.org/pdf/2304.13705)参考值来自三个训练种子、每种子 50 回合；本轮是单种子探索性结果，不能当作论文多种子的完整统计复现。

完整原始奖励、动作和 qpos 在各评估目录的 episodes.jsonl / episode_XX.npz；summary.csv 保留阶段奖励达到率、训练及评估耗时、对应检查点目录。

前一次自动启动因配置键拼写错误在进入训练前失败，现场保留在 models/cube_transfer_human_seed0/；修复后在独立 attempt1 目录从头训练，不把仅有权重当作优化器恢复点。

CPU 渲染器相对于训练数据的光栅化实现可能存在像素差异；相机过滤在同一后端验证为精确等价，不能据此推断不同图形驱动的策略成功率完全不变。

Insertion 的已有下载片段保留，自动下载已按用户要求停止；这些文件不能被认作通过检查的完整示范。

全局 finish.txt 仅在三个任务均完成 2000 轮、三组均实际评估 50 回合、150 条记录、三个图表及完整性检查通过后原子创建。当前 Cube 任务级完成记录不等于 V3 全部完成。

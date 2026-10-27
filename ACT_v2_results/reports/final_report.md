# ACT Transfer Cube V2 复现实验

全部正式实验及完整性检查完成。更新时间：2026-10-27T15:47:40.037681+08:00

## 实际协议与来源

本轮只使用官方 ACT、50 条程序化示范、顶部 480×640 RGB。用户为加快进度将正式协议改为训练种子 [0]：共 3 个模型（k=1、100、400），各训练 2000 epochs，4 组评估、每组 50 回合，总计 200 回合。k100 的同一个模型复用于时间集成主实验和无时间集成消融，不重复训练。
seed=0 按最小编号及保留已有工作的固定规则选择，不按成功率挑选。seed=1/2 已有权重、日志和部分评估保留，记录于 run_state.json 的 excluded_models / excluded_evaluations，全部排除正式统计。协议变更前文件保存在 backups/single_seed_scope_*。
固定 batch_size=8、lr=1e-5、kl_weight=10、hidden_dim=512、dim_feedforward=3200、ResNet18。保持官方 ImageNet 预训练骨干，ACT 其余参数重新初始化，不加载 E1 权重。
本地 Git：`bec6795c25dff6ddee5465bbfbe91db59ffdcc5b`；[官方代码](https://github.com/tonyzhaozh/act)固定提交：`742c753c0d4a5d87076c8f69e5628c79a8cc5488`。模型、损失、数据加载、训练/评估代码与官方逐文件相同，本机 constants 仅包含路径配置差异。
官方 main() 在加载数据前固定种子 1；正式模型共用相同的 40 条训练、10 条验证轨迹。train_bc 再设置模型训练种子。官方归一化统计覆盖所有 50 条轨迹（包含验证轨迹），保留该行为，不擅自改为训练集统计。划分见 preflight.json 和各模型 data_split.json。
评估保持官方 seed=1000，所有配置 50 回合、每回合 400 步、最高奖励等于 4 判成功；检查各配置初始方块序列一致。没有改变动力学、动作分布或奖励计算。

## EGL 与 CPU 渲染加速

**NVIDIA EGL 未启用。** 复查仍缺少 libEGL.so.1、NVIDIA vendor 和图形驱动组件，容器没有 graphics 能力且无平台管理权限。按用户最后明确授权，V2 改用 OSMesa CPU 渲染；训练和推理仍使用 A800 CUDA GPU。原 EGL 阻塞报告、管理员配置和本轮复测日志均保留。
实际 GL_RENDERER：`llvmpipe (LLVM 17.0.6, 256 bits)`，Mesa 24.0.5。库来自 Ubuntu 官方 noble 包，下载元数据和 SHA256 保存在 runtime/packages/package_metadata.json，只解包到 V2 用户目录，没有 apt 安装、系统库覆盖、驱动/CUDA 或 Conda 包变更。
Conda Python DT_RPATH 优先加载旧 OSMesa，隔离试验曾发生崩溃，日志完整保留。最终通过进程内 ctypes 库请求路由加载指定 OSMesa，并预加载其配套 GLAPI；只影响本轮进程，不修改 Python 包。真实 Transfer Cube 多相机/顶部相机验证通过才采用。
两组固定方块初始状态，各比较 reset 和五步动作：同一 LLVMpipe 内顶部图像、qpos/qvel、完整物理状态、奖励及 NumPy/Python/Torch RNG 逐位相同。优化只删除 angle、vis 两次无用渲染，AST 验证其余观测逻辑不变。
软件渲染器比较：旧 softpipe 与新 LLVMpipe 物理状态完全相同，像素 MAE 0.4015/255，RMSE 3.6891，最大差异 139，通道差异>10 占 1.365%。已查看两组 PNG，视角和物体布局正常；边缘存在光栅化差异，不要求逐像素相同。
短测中顶部路径软件渲染加速 5.54×；加上顶部相机过滤，相对旧三相机路径为 16.35×。这不是 NVIDIA EGL 加速，不代表端到端训练或全部策略回合的加速比。实际正式评估耗时逐回合记录。
参考：[Mesa LLVMpipe 文档](https://docs.mesa3d.org/drivers/llvmpipe.html)、[Ubuntu Mesa 包来源](https://archive.ubuntu.com/ubuntu/pool/main/m/mesa/)。

## 调度、记录和恢复

训练并发：1；评估并发上限：2。训练先实测 1/2/3 进程，按预热后重叠时段单位墙钟时间内实际完成的总 epoch 数选择，完整墙钟吞吐另记；最多三进程。只有 6 核 CPU 配额，不能根据 GPU 显存单独决定。评估用同一初步模型的单/双进程短测选择 1 或 2。后续按用户要求补测 100 轮的第三进程，结果见 live_parallel_tuning.json。
用户要求尽量并行后，调度器允许已完成模型的评估与剩余训练重叠（训练期间评估进程上限见 run_state.json 的 evaluations_during_training，训练完成后使用全部评估槽位）。调度器交接只替换监督进程，保留正在训练的 worker PID、优化器及所有内存状态；交接证据见 reports/parallel_scheduler_migration.json。parallel_controls.json 可在不重启任务的情况下调整并发上限；降低上限仅限制后续启动，不中断正在运行的模型。
资源记录见 logs/resource_monitor.jsonl。每个正式任务具有独立目录、日志、完整参数和源代码信息，训练每轮追加 train/validation 指标，保存最佳验证及最终检查点和配套 dataset_stats.pkl。模型权重不包含完整优化器状态；中断后保留现场，在新的 attempt 目录从头训练，不冒充无损恢复。
吞吐及冒烟模型只在 tests/ 下，排除在正式模型和全部正式统计之外。当前缩减为单种子后，保留一个尚未完成的训练进程，并允许两个已完成模型的评估并发。

## 逐种子结果

| 组别 | k | 训练种子 | TA | 成功/回合 | 成功率 | 平均 return |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| main | 100 | 0 | True | 49/50 | 98.00% | 639.92 |
| ablation | 100 | 0 | False | 46/50 | 92.00% | 613.80 |
| ablation | 1 | 0 | False | 0/50 | 0.00% | 0.00 |
| ablation | 400 | 0 | False | 48/50 | 96.00% | 490.08 |

## 正式结果汇总

| 组别 | k | 已完成种子 | 成功次数/回合 | 成功率（多种子时附样本标准差） |
| --- | ---: | ---: | ---: | ---: |
| main | 100 | 1/1 | 49/50 | 98.00%（单种子；种子间标准差不适用） |
| ablation | 1 | 1/1 | 0/50 | 0.00%（单种子；种子间标准差不适用） |
| ablation | 100 | 1/1 | 46/50 | 92.00%（单种子；种子间标准差不适用） |
| ablation | 400 | 1/1 | 48/50 | 96.00%（单种子；种子间标准差不适用） |

当前只有一个训练种子，不能估计种子间变异或样本标准差；CSV 的 sample_std_seed_success_rate 留空，不填 0，消融图不绘制种子间误差棒。50 个评估回合不等于 50 个训练种子。阶段图使用官方奖励阈值达到率；另从接触数据只读记录“左右夹爪任一接触方块”“接触方块且脱离桌面”“左夹爪接触”实际事件比例，CSV 同时给出这些指标。奖励≥2 是奖励阶段代理，不能简单等同于独立几何抬起判据。

## 与论文和历史结果的比较

[ACT 论文 Table I](https://arxiv.org/pdf/2304.13705)程序化示范 Transfer Cube 的 86% 是三个训练种子、每种子 50 回合的论文参考值，不是本轮实测。
本轮主结果为 98.00%，50 回合共成功 49 次；与论文 86% 相差 +12.00 个百分点。本轮为单训练种子结果，论文为三个训练种子的平均，统计可靠性不同，不能据此宣称完成多种子复现。训练轮数和软件渲染器等协议及环境条件也需要一并考虑。
E1 的 k100、1000 epochs、30 回合、无时间集成成功率 40% 只作历史参考，未合并到 V2 统计。

## 视频、示范核查与限制

每组只保存其实际第一个成功回合及第一个失败回合视频；若未出现某类，videos.json 明确记录缺失，不虚构案例。完整 50 回合逐步奖励和初始状态用于统计，不能用视频选择样本代替总体成功率。
示范 50 文件的 SHA256 与 E1 数据清单逐条相同，动作、qpos/qvel 有限，图像形状/类型正确，官方加载器可读，未修改 HDF5。数据没有直接保存原始初始方块 env_state，未在不能可靠恢复该状态的情况下进行专家成功率判定，也没有把错误回放当作数据失败。
CPU 光栅化与训练数据来源渲染器有输入差异；小样本状态/像素验证不能保证所有策略的成功率不受影响，应在条件允许时独立做 NVIDIA EGL 的匹配对照。

## 文件与完成状态

CSV：per_seed_results.csv、episode_results.csv、summary.csv。逐步奖励见各评估目录 episodes.jsonl / episode_XX.npz；检查点、统计和配置在 models/ 下。
图：figures/main_success.png、figures/chunk_ablation.png、figures/reward_stage_attainment.png。
完成条件以用户授权的 CPU 渲染及单训练种子协议为准；绝不创建 NVIDIA EGL 成功标志。只有 3 个模型、4 组评估、200 回合、图表和完整性检查全部通过后，才创建 V2 finish.txt。
当前 finish.txt 状态：本次最终完整性检查后创建，明确标注 OSMesa CPU。

补充配对推理诊断：用独立 30-epoch 测速模型，在 12 对 old/new 图像和相同 qpos 下，归一化动作 MAE=0.089804、最大绝对差=0.300928；反归一化后的混合动作维度平均绝对差=0.028337（各维物理量不同，不能统一解释成弧度）。这些是可测的策略输入敏感性，**不能声称切换渲染器不影响动作或成功率**。本诊断没有形成成功率估计，不纳入 V2 正式统计。详见 software_renderer_policy_sensitivity.json。

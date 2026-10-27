# Insertion 数据整理与校验

时间：2026-10-27T14:41:17.156875+08:00

| 数据集 | 文件数 | 轨迹长度 | 原始 ZIP | 官方加载器 | 就绪 |
| --- | ---: | ---: | ---: | --- | --- |
| sim_insertion_scripted | 50 | 400 | 10 | 通过：train/validation 各一个 batch=8 | True |
| sim_insertion_human | 50 | 500 | 12 | 通过：train/validation 各一个 batch=8 | True |

逐个普通 ZIP 和全部成员流式 CRC 校验；校验重复编号 SHA256，完整检查每条 HDF5 的所有图像、qpos/qvel/action、NaN/Inf、sim 标志和字段形状。
Human 使用 500 步，Scripted 使用 400 步，均为 top RGB (480,640,3) uint8；动作与状态每步 14 维。
官方 utils.load_data 实际读取独立数据集，固定划分种子 1，40 条训练 / 10 条验证；CPU 执行，CUDA_VISIBLE_DEVICES 为空。分别生成各自归一化统计，不跨数据集使用。
原始 ZIP 名称、大小及 SHA256 保留验证：True。没有覆盖任何已有 HDF5，没有修改 constants.py、V1/V2 结果或正在运行的 Cube 任务。
数据校验就绪不等于专家成功率已验证；未做专家回放，不声称示范成功率 100%。
本次只整理数据；Insertion 训练开关仍保持关闭，未自动启动新训练或仿真评估。
详细清单：archive_inventory.json、archive_members.csv、dataset_manifest.csv、preparation_results.json。

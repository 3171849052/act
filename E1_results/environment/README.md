# 本次环境清单

这些文件是已配置环境的版本记录，复制自原服务器 results/，未重新安装任何依赖。
实际核心版本为 Python 3.8.10、Torch 2.0.1+cu118、TorchVision 0.15.2+cu118、MuJoCo 2.3.7、dm-control 1.0.14、h5py 3.10.0。

当前激活方式仅适用于原服务器：在项目根目录执行 `source results/activate_act_repro.sh`。results/ 的安装缓存不在发布范围，其他机器应按版本清单准备独立环境，并单独验证离屏渲染。

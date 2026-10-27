# NVIDIA EGL 离屏渲染实际验证报告

检查日期：2026-10-27（Asia/Shanghai）。所有结果位于当前目录。

**结论：NVIDIA EGL GPU 渲染未跑通，阻塞于容器的 EGL 动态库/图形驱动组件暴露层。需要平台管理员在新验证容器中补齐图形组件。没有创建 `egl_success.txt`。**

## 1. 当前服务器与容器环境

- 系统 Ubuntu 24.04.1 LTS，普通用户 UID 11519、GID 9137。
- `/proc/1/cgroup` 为 `0::/`，单独不足以辨认容器类型。
- `mountinfo.txt` 同时包含 kubelet Pod 路径、containerd CRI sandbox 路径和 containerd overlay snapshot，支持 Kubernetes/containerd 容器的判断。
- `/.dockerenv`、`/run/.containerenv` 均不存在；没有发现 Docker/Podman 管理 socket。
- Docker、Podman、Apptainer、Singularity、kubectl、ctr、crictl、nvidia-container-cli 均不在 PATH。无法从内部确认命名空间、上层控制器、具体 runtime handler 或 legacy/CDI 模式，未猜测这些信息。
- `NVIDIA_VISIBLE_DEVICES=all`；`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`；`CUDA_VISIBLE_DEVICES` 未设置。
- 证据：`runtime_environment.json`、`cgroup.txt`、`mountinfo.txt`、`os_release.txt`。

## 2. GPU、驱动与设备权限

- 实际可见 GPU：NVIDIA A800 80GB PCIe，81920 MiB，逻辑编号 0。
- 驱动：595.45.04；PCI Bus：00000000:A4:00.0；UUID：GPU-13c992de-1d74-d12c-bd2b-9e4b8e09d6b3。
- 检查时显存使用 0 MiB、GPU 利用率 0%，nvidia-smi 未列出运行进程；MIG Disabled，未发现 nvidia-smi 通信错误。
- nvidia-smi 显示 CUDA Version 13.2，这是驱动支持能力，不能替代 ACT 的 PyTorch CUDA 构建版本记录。
- 实际节点是 `/dev/nvidia7`、`/dev/nvidiactl`、`/dev/nvidia-uvm`、`/dev/nvidia-uvm-tools`，均为 `crw-rw-rw-`，当前用户 `os.access` 检查可读写。
- `/dev/dri` 不存在。当前错误发生在加载 EGL 库之前，尚不能将其归因于设备权限或缺少 DRM render node。未创建节点、未修改权限。
- 已保存完整 `nvidia_smi.txt`、`nvidia_smi_q.txt`、`gpu_identity.csv`、`device_nodes.txt`。

## 3. EGL 动态库和 vendor 检查

检查使用 `ldconfig -p`、`find`、realpath/readlink、`ldd`、`ctypes.util.find_library`、`ctypes.CDLL`。搜索范围为 `/usr`、`/opt`、`/lib`、`/run/nvidia` 及整个 act-repro 前缀；`/run/nvidia` 不存在，find 退出码 1 的原因已保留。没有遍历不可见的宿主机文件系统。

| 组件 | 实际结果 |
| --- | --- |
| `libEGL.so` / `libEGL.so.1` | 上述搜索范围未找到；原 ACT 库路径及 Conda 完整 lib 路径下均无法加载 |
| `libEGL_nvidia.so.*` | 未找到，CDLL 加载失败 |
| `libnvidia-eglcore.so.*` | 未找到 |
| `libGLX_nvidia.so.*` | 未找到，CDLL 加载失败 |
| `libGLdispatch.so.0` | Conda 内存在，ACT 隔离路径可加载，其 ldd 依赖无缺失 |
| `libOpenGL.so.0` | Conda 内存在，ACT 隔离路径可加载，其 ldd 依赖无缺失 |
| 系统 EGL vendor 目录 | `/usr/share/glvnd/egl_vendor.d/`、`/etc/glvnd/egl_vendor.d/` 均不存在 |
| NVIDIA vendor JSON | 在检查范围内未发现 |
| Mesa EGL | Conda 编译 sysroot 中有 `libEGL_mesa.so.0` 和 `50_mesa.json`，不是 NVIDIA vendor，也不是通用 `libEGL.so.1` |

Mesa JSON 引用 `libEGL_mesa.so.0`，sysroot 对应文件存在；按绝对路径加载仍失败，报 `libxshmfence.so.1` 缺失。`ldd` 另列出多项 XCB/Wayland/GBM 等依赖缺失；这些是该 Mesa 副本的依赖问题，不能据此推断 NVIDIA EGL 的依赖状态。没有把它重命名或链接成 NVIDIA EGL，也没有把 sysroot 全局加入搜索路径。

证据：`ldconfig.txt`、`library_search.txt`、`library_dependencies.json`、`egl_vendors.json`、`vendor_reference_loading.json`、`python_dependencies_and_loading.txt`。

## 4. Python 依赖实际版本

使用 `/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs/act-repro/bin/python`，没有使用系统 Python 进行渲染。

| 包 | 版本 |
| --- | --- |
| Python | 3.8.10 |
| MuJoCo | 2.3.7 |
| dm-control | 1.0.14 |
| PyOpenGL | 3.1.10 |
| PyOpenGL-accelerate | 未安装 |
| glfw | 2.11.0 |
| NumPy | 1.23.5 |
| Pillow | 10.4.0 |
| PyTorch | 2.0.1+cu118（仅读取包版本，没有启动模型） |

未发现证据表明应升级 Python 包。缺少原生 EGL/NVIDIA vendor 库无法通过升级 PyOpenGL 解决，未克隆、重装或升级环境。

## 5. eglQueryString 错误的确认原因

独立 `probe_egl.py` 在导入前设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`MUJOCO_EGL_DEVICE_ID=0`、`MPLBACKEND=Agg`。
实际在 `import mujoco` 时，经 `mujoco.egl` → `OpenGL.EGL` → `OpenGL.raw.EGL._types` 复现：

```text
raw_eglQueryString = _p.PLATFORM.EGL.eglQueryString
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

原生 `ctypes.CDLL('libEGL.so.1')` 同时报 `cannot open shared object file: No such file or directory`。
额外在新进程中仅把 `LD_LIBRARY_PATH` 换成 Conda 完整 lib，直接检查：

```text
platform = OpenGL.platform.egl.EGLPlatform
PLATFORM.EGL = None
find_library(EGL) = None
```

该进程 `from dm_control import mujoco` 也经 dm_control._render 复现同一异常。
安装版本的 `OpenGL/platform/ctypesloader.py` 在尝试 libEGL 名称均失败后返回 None；`_types.py` 随后访问其 eglQueryString 属性。**这是原生库没有加载成功的错误，不是已经调用 eglQueryString 后失败，更不是无上下文 glGetString 的误判。**

NVIDIA 官方说明 graphics 能力用于 OpenGL/EGL/Vulkan，能力变量控制容器中的驱动组件挂载。结合实测缺少 graphics、缺少 NVIDIA 图形库与 vendor，可确认当前容器未提供所需图形栈。仅凭容器内部证据，仍不能区分宿主机缺少对应用户态组件与 runtime 未注入组件，需管理员检查宿主机和创建模板。[NVIDIA 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)

完整错误：`egl_probe.log`、`egl_probe_result.json`、`egl_loader_isolation.log`。

## 6. 实际采取的安全措施与修复尝试

1. 用项目原 `activate_act_repro.sh` 的隔离库路径运行独立 EGL 探针。
2. 在另一个进程临时改用 Conda 完整 lib，排除只因 `lib/act-graphics` 屏蔽已有 EGL 库的可能；结果仍失败。
3. 检查已有 vendor JSON 引用、共享库真实目标、依赖和实际加载；没有可供用户态 JSON/搜索路径修复使用的 NVIDIA vendor 库。
4. 保持原 OSMesa 配置不变，在隔离进程完成真实 ACT 环境的短基线。
5. 将需要平台参与的步骤停止在说明阶段，没有尝试提权、修改系统库或重新创建容器。

没有执行 sudo、apt、驱动/CUDA 修改、包安装、全局链接修改、容器平台变更或杀死其他进程。没有通过在现有容器中 export graphics 宣称修复。

## 7. NVIDIA EGL 与 Transfer Cube GPU 验证状态

| 验收项 | 实际结果 |
| --- | --- |
| 创建 NVIDIA EGL 上下文 | 失败，导入期原生库加载阻塞 |
| EGL GL_VENDOR / GL_RENDERER / GL_VERSION | 均为 null，未创建有效上下文，因此未调用 glGetString |
| 简单 EGL MuJoCo RGB | 未执行到渲染阶段，没有 EGL PNG |
| ACT dm_control Transfer Cube GPU reset/step | 按前置条件跳过，`gpu_render_test.log` 明确记为 preflight blocked |
| NVIDIA GPU 实际启用 | 未验证成功 |
| CPU/GPU 物理状态、奖励、图像对比 | 尚不能执行，JSON 均为 null |
| 成功标志 | 未创建 `egl_success.txt` |

`probe_egl.py` 与 `test_transfer_cube_egl.py` 的查询函数会在 **physics.contexts.gl.make_current()** 中，通过其 executor.call 在上下文所属线程调用 glGetString，记录的是 dm_control 自身渲染上下文。GPU 分支要求 NVIDIA 厂商、EGL context class，排除 llvmpipe/softpipe/swrast/software。已按安装版 API 编写并检查 Python 3.8 语法；受当前阻塞影响，GPU 成功分支尚未实际验证。

## 8. OSMesa 真实环境基线与性能

本次只运行短程渲染验证，不加载 ACT 模型，不训练，不做完整评估。调用未经修改的
`sim_env.make_sim_env('sim_transfer_cube_scripted')`，固定 BOX_POSE 为
`[0.15, 0.5, 0.05, 1, 0, 0, 0]`，动作从起始 14 维观测构造，两个腰关节施加小幅确定性运动；实际转换后的 16 维控制量均通过 actuator_ctrlrange 合法性检查。

先预热两次 reset、每次两个 step 和一次顶部渲染；再测三次 reset、每次五步，共 15 个 step 和 15 次单独 top render。
保持 sim_env 原始行为：**reset/step 包含 top、angle、vis 三个相机观测；单独 render_top 只测 top 相机的 480×640 RGB。** 两类时间不能混作单帧渲染时间。

OSMesa 实际身份：

```text
GL_VENDOR   = Mesa/X.org
GL_RENDERER = softpipe
GL_VERSION  = 3.3 (Compatibility Profile) Mesa 21.2.5
context     = dm_control._render.pyopengl.osmesa_renderer.OSMesaContext
```

| 指标 | OSMesa 实测 | NVIDIA EGL |
| --- | --- | --- |
| reset 平均耗时（3 次） | 1908.19 ms | 不可测 |
| step 平均耗时（15 次） | 1913.50 ms | 不可测 |
| top render 平均耗时（15 次） | 647.46 ms | 不可测 |
| top render 标准差 | 4.94 ms | 不可测 |
| 单独 top render 吞吐 | 1.5445 帧/秒 | 不可测 |
| 环境 step 吞吐（只含 step 耗时） | 0.5226 步/秒 | 不可测 |
| 总测量段吞吐 | 0.3372 步/秒 | 不可测 |
| 实测加速比 | 无法计算，null | 无法计算，null |

总测量段 44.4905 秒，包括 reset、step、额外单相机 render、状态/图像采集和首次 PNG 保存；不含导入、预热和最后的 NPZ 压缩保存。结果只代表本次短基线，不能外推成正式 ACT 训练速度。

CPU 图像为 `(480,640,3)`、uint8、RGB。保存了 reset/final PNG 和 18 个时刻的状态、控制、奖励及顶部图像（NPZ，不涉及 HDF5）。reset 图已实际查看，可见双臂、桌面和红色方块；没有 GPU 对照图，不能声称跨渲染器视觉一致。
全部采集状态/奖励有限，MuJoCo 警告计数全为零，当前上下文 glGetError=0，physics.free 成功释放上下文。这里的稳定性仅为 CPU 侧结果。

证据：`osmesa_render_test.log`、`osmesa_transfer_result.json`、`osmesa_trajectory.npz`、`osmesa_transfer_top_reset.png`、`osmesa_transfer_top_final.png`、`benchmark_results.json`。

`benchmark_render.py` 已实际启动两个独立子进程：OSMesa 完成，EGL 子进程在前置探针检查处退出，未执行 GPU Transfer Cube。未来两者均通过后，会计算 qpos/qvel、观测、env_state、控制、仿真时间、奖励和 timestep 的误差，并记录图像 MAE/RMSE、通道误差与差异比例；不会强制 RGB 逐位相同，仍要求人工查看 PNG。

## 9. 对已有 ACT 环境和结果的影响

所有新增文件仅写入 `ACT_v2_results/egl_validation/`。未修改 sim_env、constants、ACT 模型/训练算法、激活脚本、HDF5 或 E1_results 历史结果，未启动训练或模型评估。
对 Git 已跟踪文件及 Conda 包元数据/历史做了前后 SHA256 对比，`environment_integrity.json` 显示无变化；这不是对整个 Conda 前缀所有文件的全量取证，但本次没有进行依赖安装或环境文件编辑。
原 ACT 默认仍为 OSMesa；EGL 环境设置仅存在于相应测试进程。

## 10. 管理员需要采取的操作

详见 [管理员图形配置与重测办法](admin_graphics_configuration.md)。核心是复制原 Kubernetes 任务为验证任务，在创建前启用 `compute,utility,video,graphics`，确认现有 NVIDIA containerd runtime/设备注入方案实际带入匹配 595.45.04 的 NVIDIA 图形库、通用 GLVND EGL 入口及 NVIDIA vendor JSON。保留数据挂载、原环境和平台 GPU 分配，不修改宿主驱动/CUDA，也不要求 privileged。

只在存在 NVIDIA vendor 库而 JSON 搜索路径错误时，才适合用户目录 JSON 或 `__EGL_VENDOR_LIBRARY_FILENAMES` 修复；当前缺失库，配置文件本身不足以修复。仅安装通用 libEGL 也不足以让 A800 EGL 工作。

该管理员文档提供原 Pod 模板合并片段、只读 kubectl 查询、库验收与重新运行命令，未执行任何集群写操作。参考：[NVIDIA containerd 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-containerd-for-kubernetes)、[NVIDIA Kubernetes GPU 任务](https://github.com/NVIDIA/k8s-device-plugin#running-gpu-jobs)。

## 11. 后续正式仿真的环境变量与启动方式

当前尚不能用于正式 EGL 仿真。管理员补齐图形组件、归档本次证据并通过探针和对比后，在项目根目录、**导入 MuJoCo/dm_control/OpenGL 之前**执行：

```bash
source results/activate_act_repro.sh
# 激活脚本会设为 osmesa，因此以下覆盖必须在 source 之后。
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export MPLBACKEND=Agg
python -u ACT_v2_results/egl_validation/probe_egl.py
# 仅 NVIDIA 探针通过后：
python -u ACT_v2_results/egl_validation/benchmark_render.py
```

正式仿真命令应继承这组变量；本次没有运行训练/评估命令。`MUJOCO_EGL_DEVICE_ID=0` 指 EGL 枚举编号，最终仍须由 GL_RENDERER 确认新容器实际所选 GPU。
`NVIDIA_DRIVER_CAPABILITIES` 的 graphics 必须由平台在创建时配置，不能用上面的 shell export 代替挂载操作。
在修复完成之前，现有可用仿真继续使用原激活脚本的 `MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa` 和 `MPLBACKEND=Agg`。

## 12. 文件与完成标准

- 入口脚本：`probe_egl.py`、`test_transfer_cube_egl.py`、`benchmark_render.py`。
- 原始环境采集：`collect_environment.py` 及对应 txt/json 日志。
- 运行日志：`egl_probe.log`、`gpu_render_test.log`、`osmesa_render_test.log`、`benchmark.log`。
- 状态与证据：`egl_probe_result.json`、`egl_transfer_result.json`、`osmesa_transfer_result.json`、`benchmark_results.json`、`environment_integrity.json`。
- 当前状态：`egl_status.txt`、`egl_blocked.txt`。
- 尚未满足 EGL 初始化、NVIDIA 身份、GPU Transfer Cube reset/step/图像和 CPU/GPU 比较等条件，**未创建成功标志，不提供虚构的 GL_RENDERER 或加速倍数**。

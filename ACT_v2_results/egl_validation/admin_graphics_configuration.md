# 管理员操作说明（未执行任何平台变更）

## 当前阻塞与定位

当前任务位于 Kubernetes 管理的 containerd 容器：`mountinfo.txt` 同时含
`/var/lib/kubelet/pods/…`、`io.containerd.grpc.v1.cri/sandboxes/…` 和 containerd overlay snapshot。
这是挂载证据支持的判断；无法从容器内部确定集群命名空间、控制器类型、NVIDIA runtime 的 legacy/CDI 模式及平台的具体配置策略。
当前没有 docker/podman/apptainer/singularity/kubectl/ctr/crictl/nvidia-container-cli 或容器管理 socket。
不要在这个容器里执行 Docker 重建命令。

设备 `/dev/nvidia7` 对当前用户可读写，nvidia-smi 将它显示为逻辑 GPU 0：
NVIDIA A800 80GB PCIe，UUID `GPU-13c992de-1d74-d12c-bd2b-9e4b8e09d6b3`，驱动 `595.45.04`。
不能因此假定存在 `/dev/nvidia0`，也不能把主机设备编号 7 直接用作 EGL 枚举编号。

确定的缺失包括：

1. 创建时环境为 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`，缺少 `graphics`。
2. 原有隔离库路径及 Conda 完整 lib 路径下均无法加载 `libEGL.so.1`。
3. 系统/Conda 检查范围没有 `libEGL_nvidia.so.*`、`libnvidia-eglcore.so.*`、`libGLX_nvidia.so.*`，也没有 NVIDIA vendor JSON。
4. PyOpenGL 平台选择正确为 EGL，但 `PLATFORM.EGL=None`，在调用 `eglQueryString` 前失败。

## 对当前 Kubernetes/containerd 平台的最小变更

请在平台控制台找到此交互任务对应的原始工作负载模板。保留原镜像、用户、GPU 分配、数据卷、工作目录、网络和已有资源限制，复制为一个验证任务。在**新容器创建前**将其容器环境变量改为：

```yaml
# 合并到原 Pod spec.containers 中相应容器的 env；不是完整可 apply 的资源。
env:
  - name: NVIDIA_DRIVER_CAPABILITIES
    value: compute,utility,video,graphics
```

保留平台原有 GPU 资源申请方式；若使用标准 NVIDIA device plugin，其 Pod 容器资源通常包括：

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

不要手工设置 `NVIDIA_VISIBLE_DEVICES=all` 来绕过平台设备分配。
若 NVIDIA runtime 不是默认 runtime，使用集群**已经配置且验证可用**的 NVIDIA RuntimeClass；其名字需管理员查询，不能假定必为 `nvidia`。
若平台使用 CDI 或自定义设备注入，确认相应注入配置把 OpenGL/EGL 图形驱动组件一并传入，不能仅依赖 env 存在。
上面的 env 在运行中的 shell 里 export 不会增加驱动库挂载。

如果需要定位工作负载，可在管理员终端只读查询（将占位符替换为平台实际值）：

```bash
kubectl -n <namespace> get pod <pod-name> -o yaml
kubectl -n <namespace> get pod <pod-name> -o jsonpath='{.metadata.ownerReferences}'
kubectl get runtimeclass
```

节点已有 CUDA GPU 可用，不能据此要求重装驱动或重配整个集群。先检查现有 NVIDIA Toolkit/runtime 的图形组件注入。
如果宿主机确实未提供与 **595.45.04** 匹配的用户态 NVIDIA EGL 组件，由管理员/镜像维护者按平台驱动挂载方案补齐；本任务不修改宿主 NVIDIA 驱动或 CUDA。
如果新验证镜像缺少通用 GLVND EGL 入口，镜像维护者另行提供 `libEGL.so.1` 和它的依赖，不能用 Mesa vendor 库重命名代替。

## 新容器中应具备的库和配置

必须实际可加载，而非只有文件名：

- 通用 GLVND：`libEGL.so.1`、`libGLdispatch.so.0`、可用的 OpenGL dispatch 库。
- 匹配宿主版本的 NVIDIA vendor：`libEGL_nvidia.so.0`、`libnvidia-eglcore.so.595.45.04` 及其实际 ldd 依赖；检查 `libGLX_nvidia.so.*` 的图形组件注入。
- NVIDIA vendor JSON，内容的 `ICD.library_path` 指向实际存在且可加载的 NVIDIA EGL 库。例如：

```json
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
```

文件通常位于 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`，但最终以平台真实安装位置为准。
若只有 JSON 搜索路径不匹配，可以在探针进程中设置
`__EGL_VENDOR_LIBRARY_FILENAMES=/实际路径/10_nvidia.json`；当前没有 vendor 库，创建这个 JSON 没有修复作用。
每个库运行 `readlink -f`、`ldd`，确认无 `not found`，再在 act-repro Python 内 `ctypes.CDLL` 验证。
只有初始化实际返回 NVIDIA GL_VENDOR/GL_RENDERER 才算图形栈可用。

`/dev/dri` 当前不存在，但这本身不能证明是 NVIDIA EGL 失败的原因；当前故障在加载 EGL 库之前，尚未进入设备枚举或权限验证。
若补齐库后出现设备访问错误，再依据具体 errno 和设备路径处理，不能先要求添加所有设备或启用 privileged。

## 修复后的独立复测

在项目根目录执行，测试不训练、不加载 ACT 模型：

```bash
source results/activate_act_repro.sh
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/ACT_v2_results/egl_validation/matplotlib"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export MPLBACKEND=Agg
python -u ACT_v2_results/egl_validation/probe_egl.py > ACT_v2_results/egl_validation/egl_probe.log 2>&1
# 仅当前一命令退出码为 0、JSON passed=true 且真实厂商为 NVIDIA 后继续：
python -u ACT_v2_results/egl_validation/benchmark_render.py > ACT_v2_results/egl_validation/benchmark.log 2>&1
```

这里选择的是 EGL 枚举设备 0；新容器的 NVIDIA EGL 设备枚举必须通过探针确认。
benchmark 会生成独立的 CPU/GPU 子进程，两者均使用原始 `sim_env.make_sim_env`、固定方块位置、相同 14 维动作序列、相同分辨率。
`reset/step` 保持 ACT 原来三个相机观测；单独 `render_top` 测量顶部相机。
对比状态和图像后必须检查保存 PNG 的视角、物体与颜色。脚本**不会自动创建成功标志**；全部验收条件成立后才可创建。
在重新执行前请归档本次诊断目录，保留现有失败证据。

## 官方依据

- [NVIDIA Driver Capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)：`graphics` 对应 OpenGL/EGL/Vulkan 的驱动能力；环境变量控制创建容器时挂载的驱动组件。
- [NVIDIA Toolkit：containerd for Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-containerd-for-kubernetes)：containerd 的 NVIDIA runtime 配置由节点/平台管理员负责。本次未执行该文档中的节点修改或重启。
- [NVIDIA Kubernetes device plugin](https://github.com/NVIDIA/k8s-device-plugin#running-gpu-jobs)：标准 GPU 资源申请及 RuntimeClass 的说明。平台可能有自己的调度包装，以上片段应合并到实际模板。

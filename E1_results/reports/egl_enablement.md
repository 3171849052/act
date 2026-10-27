# EGL 硬件渲染启用条件与实际检查

本次已在独立进程中实际重测 EGL；失败日志见 `egl_acceleration_probe.json` 与 `../logs/egl_acceleration_probe.retry1.*.log`。
未修改正在运行的评估进程环境，未安装或重新安装依赖，未修改系统 CUDA、NVIDIA 驱动或 Python。

实测限制：

- 容器 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`，没有 `graphics`。
- 当前搜索路径未发现 `libEGL.so.1`、`libEGL_nvidia.so.*`、`libnvidia-eglcore.so.*`、`libGLX_nvidia.so.*`。
- Docker 二进制、`/var/run/docker.sock` 和 NVIDIA 容器管理命令均不可用，没有可用的容器启动配置管理入口。
- EGL 探针实际在导入 OpenGL EGL 时失败：`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`。

因此，当前会话无法完成 GPU EGL 启用。不能靠在已运行容器里单独 export 环境变量补出缺失的挂载库。
NVIDIA 官方说明：`graphics` 能力负责 OpenGL/EGL/Vulkan 的驱动库暴露：
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities

可交给平台管理员的具体配置：

1. 在新容器/任务的创建配置中保留原 GPU、数据盘与工作目录挂载，将 `NVIDIA_DRIVER_CAPABILITIES` 设为 `compute,utility,video,graphics`，由 NVIDIA Container Toolkit 暴露与宿主驱动匹配的图形库。
2. 使用现有 act-repro 环境；不更换系统 CUDA、驱动、系统 Python，也不重新安装项目依赖。
3. 仅对独立探针设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`，确认实际 GL_VENDOR/GL_RENDERER 为 NVIDIA A800，而不是 Mesa 软件回退。当前图形激活脚本设置 osmesa，测试时需在 source 后覆盖。
4. 先执行环境 reset/step、顶部图像、固定轨迹及模型推理测速，对比图像与动作/奖励；不同渲染后端可能有像素差异，不把局部 EGL 回合混入现有 OSMesa 统计。
5. 验证通过后，若要本次正式结果改用 EGL，应在新目录统一重跑四组各 30 回合并保留旧结果，报告明确记录实际后端。

目前正式评估继续使用已验证的 OSMesa、顶部相机过滤、最多三个独立进程，EGL 外部限制不会阻断这两项加速。

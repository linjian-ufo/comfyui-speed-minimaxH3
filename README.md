# comfyui-speed-minimaxH3

**简体中文** | [English](README_EN.md)

MiniMax H3 专用的 ComfyUI 安全缓存加速节点。它在相邻采样步骤变化较小时复用整段 Transformer 残差，以减少重复计算；同时提供显存/内存预算、CPU 自动降级、重复 sigma 隔离和缓存冲突检测。

> [!IMPORTANT]
> **三个最重要的参数在节点里以 `★` 标记：缓存复用阈值、加速区间、最大连续跳步数。** 这些参数直接影响速度和画质。鼠标悬停在参数名称或帮助图标上，可查看作用、默认值、范围、步长、调高/调低的影响与建议值。

> [!WARNING]
> 本节点仅支持 `MiniMaxH3Model`。不要与 EasyCache、`TE-Speed-MiniMaxH3`、`ComfyUI-MiniMaxH3-Cache` 或其他整模型/整 Block 缓存节点串联。

## 主要特点

- 默认值来自长视频连续运行测试，兼顾速度和画质：阈值 `0.12`、区间 `10%–90%`、最多连续跳过 `2` 步。
- 只保存少量采样特征作为变化签名，不复制整份隐藏状态来计算签名。
- `auto` 根据实际可用显存、系统内存和预留空间选择 GPU、CPU 或停用该步缓存。
- CUDA 分配失败时可转入 CPU 缓存路径，降低直接 OOM 的概率。
- 同一 sigma 的重复调用执行原模型，避免不同 CFG 条件之间错误复用残差。
- 自动检测已有 `block_loop` 缓存并拒绝串联。
- 不覆盖磁盘上的 ComfyUI 核心文件；旧版 ComfyUI 仅在运行内存中安装兼容钩子。
- 不调用 NVML 或 `nvidia-smi`，不修改显卡频率、功耗、电压或风扇。
- 支持 ComfyUI 原生中英文界面翻译和双语悬停说明。

## 安装

### Git 安装

在 `ComfyUI/custom_nodes` 目录执行：

```bash
git clone https://github.com/linjian-ufo/comfyui-speed-minimaxH3.git
```

### 手动安装

下载仓库并解压到：

```text
ComfyUI/custom_nodes/comfyui-speed-minimaxH3
```

确认该目录中可以直接看到 `__init__.py`，不要形成双层同名目录。随后重启 ComfyUI，并在节点列表搜索：

```text
MiniMax H3 Speed Cache
```

也可以从 `MiniMaxH3 → optimization` 分类添加节点。

## 使用方法

1. 加载 MiniMax H3 模型。
2. 将模型加载节点的 `MODEL` 输出连接到本节点的 `model` 输入。
3. 将本节点的 `MODEL` 输出连接到原本接收模型的采样器。
4. 第一次建议使用默认值，并打开 `verbose` 对照控制台中的 `RUN/SKIP` 统计。
5. 使用相同提示词、种子、分辨率和步数，与不使用缓存的结果比较后再调整参数。

如果鼠标悬停没有显示说明，请先升级 ComfyUI 前端并按 `Ctrl+F5` 强制刷新页面。英文界面会读取 `locales/en`，简体中文界面会读取 `locales/zh`。

## 默认参数

这组默认值经过长视频连续运行验证，适合作为首次使用的起点，但不同显卡、模型版本、分辨率、帧数和提示词仍可能得到不同结果。

```text
reuse_threshold          0.12
start_percent            0.10
end_percent              0.90
max_consecutive_skips    2
cache_device             auto
vram_reserve_gb          2.0
ram_reserve_gb           4.0
signature_tokens         128
signature_features       64
verbose                  false
```

## 参数详解

| 参数 | 默认值 | 范围 / 选项 | 作用与调节建议 |
|---|---:|---|---|
| **★ `reuse_threshold`** | **0.12** | 0.00–1.00，步长 0.005 | 核心速度/画质参数。越高越容易复用缓存、可能越快，但细节、运动和音画一致性风险越高；设为 0.00 时不跳步。 |
| **★ `start_percent`** | **0.10** | 0.00–1.00，步长 0.01 | 从采样进度的哪个位置开始允许加速。越早可能越快，但更容易影响早期结构。必须小于 `end_percent`。 |
| **★ `end_percent`** | **0.90** | 0.00–1.00，步长 0.01 | 到哪个采样位置停止跳步。越晚可能越快，但末段纹理、边缘、动作和音频细节风险更高。 |
| **★ `max_consecutive_skips`** | **2** | 1–5，步长 1 | 连续复用达到该次数后强制完整计算一次。画质优先使用 1；未经对比测试不建议超过 2。 |
| `cache_device` | auto | auto / gpu / cpu | `auto` 自动选择；`gpu` 通常最快但占用更多显存；`cpu` 更省显存但传输可能降低速度。 |
| `vram_reserve_gb` | 2.0 | 0.5–16.0 GB，步长 0.5 | auto 模式至少为模型、CUDA 工作区和 VAE 预留的显存。出现 OOM 时调高。 |
| `ram_reserve_gb` | 4.0 | 1.0–32.0 GB，步长 1.0 | auto 模式至少为系统保留的内存。系统内存紧张时调高。 |
| `signature_tokens` | 128 | 32–512，步长 32 | 变化检测采样的位置数。默认通常足够；提高会略增检测开销。 |
| `signature_features` | 64 | 16–256，步长 16 | 每个采样位置观察的隐藏通道数。默认通常足够；过低可能降低判断可靠性。 |
| `verbose` | false | true / false | 输出每步 `RUN/SKIP` 原因和最终统计。首次测试或排错时开启。 |

## 可选参数方案

### 画质优先

```text
reuse_threshold          0.08
start_percent            0.20
end_percent              0.80
max_consecutive_skips    1
cache_device             auto
```

### 均衡

```text
reuse_threshold          0.10
start_percent            0.15
end_percent              0.90
max_consecutive_skips    2
cache_device             auto
```

最高画质基准始终是不使用缓存节点。更激进的设置可能降低快速运动、细节、音画同步和长视频的一致性。

## 安全边界

本节点不会控制显卡硬件参数，也不会绕过 NVIDIA 驱动的温度或功耗保护。缓存策略仍会增加 GPU/CPU 内存占用，并可能引起 OOM、生成失败或画质变化。显卡高负载、温度、机箱散热、供电和硬件改装需要用户自行监控。

## 兼容性

- 面向具有 `MiniMaxH3Model` 的 ComfyUI 版本开发，并在 ComfyUI v0.30.0 接口上完成兼容验证。
- 新版 ComfyUI 如果原生提供 `block_loop` 接口，会直接复用。
- 旧版 ComfyUI 没有 `MiniMaxH3Model` 时会明确报错并要求升级。
- 不支持其他视频模型、图片模型或文本编码器模型。

## 常见问题

### 节点列表中找不到

- 确认文件夹不是 `comfyui-speed-minimaxH3/comfyui-speed-minimaxH3` 双层结构。
- 确认根目录中直接存在 `__init__.py`。
- 查看 ComfyUI 启动日志中是否有导入错误。
- 重启 ComfyUI，并在浏览器中按 `Ctrl+F5`。

### 提示已有 block_loop 缓存

工作流中已经存在其他整模型/整 Block 缓存。请删除其他缓存节点并重启 ComfyUI，不要把多个缓存方案串联。

### 显存不足

优先保留 `cache_device=auto`，适当提高 `vram_reserve_gb`。如果仍然不足，可尝试 `cache_device=cpu`，但速度可能下降。

### 速度提高但画质变化

依次降低 `reuse_threshold`、把 `max_consecutive_skips` 改为 `1`，并缩小 `start_percent` 到 `end_percent` 之间的加速区间。

## tests 文件夹是否必须

`tests` **不是 ComfyUI 运行插件所必需**，删除它也不会影响节点加载；但公开仓库建议保留。它验证缓存复用、重复 sigma 隔离、尺寸变化后缓存失效、阈值关闭、默认参数、节点克隆和缓存冲突检测。`__pycache__` 只是 Python 编译缓存，已通过 `.gitignore` 排除，不应上传。

在能够导入 ComfyUI 和 PyTorch 的 Python 环境中运行：

```bash
python -m unittest discover -s tests -v
```

## 许可证与来源

本项目以 [GPL-3.0](LICENSE) 发布。设计来源和第三方致谢见 [NOTICE.md](NOTICE.md)。项目不包含编译后的 `nodes.pyd`，也不会复制或替换 ComfyUI 磁盘上的 `comfy/ldm/minimax/model.py`。

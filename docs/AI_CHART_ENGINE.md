# BeatForge AI Chart Engine

AI Chart Engine 是 BeatForge Studio 的全本地五轨制谱工作流。它直接读取本地授权语料中
`SPEED_CLUB`、`SPEED_DEVIL` 与 `SPEED_REMIX` 的原始 `.sm` 和配套音频，以现有
BeatForge 声学候选为时间锚点，生成、优化、验证并导出可玩的 `pump-single` 谱面。

运行时不调用 LLM 或云端 API。语料、特征、模型 checkpoint 和生成结果全部保存在本机，
且 `local-data/`、兼容旧目录 `材料/` 与 `storage/` 都不应提交到 Git。

## 本地参考语料

默认语料目录是：

```text
local-data/speed-corpus/
├── SPEED_CLUB/
├── SPEED_DEVIL/
└── SPEED_REMIX/
```

谱面、模式和歌曲数量取决于开发者自行配置并获授权的本地语料。解析器保留变速 BPM 表、
Offset、任意行细分、Tap、Hold 与 Mine，并能恢复源文件中出现的孤立 Hold tail。

如果语料在其他位置，通过 `.env` 指定：

```dotenv
BEATFORGE_SPEED_CHARTS_DIR=/absolute/path/to/licensed-speed-corpus
```

检查解析后的语料统计：

```bash
.venv/bin/python scripts/chart_engine.py inventory
```

## 本地训练数据集

模型只训练五轨谱面。下面的命令对语料中的 `pump-single` 谱面逐首运行生产版 BeatForge
分析器，并在 `storage/chart-engine/dataset/` 生成完整训练三元组：

```bash
.venv/bin/python scripts/chart_engine.py build-dataset \
  --mode pump-single \
  --analysis-mode balanced \
  --analyze-missing
```

首次运行会分析真实音频，耗时取决于 CPU/GPU；以后会按音频 SHA-256 复用
`.feature-cache/`。同一音频的不同谱面使用相同的 train/validation/test split，避免音频泄漏。
未带 `--analyze-missing` 时，缺少真实分析的样本会被明确记录为跳过，不会生成空候选或占位特征。
如果 MP3 能读取元数据但完整 libsndfile 解码失败，构建器会自动使用本地 FFmpeg 解码为临时
PCM 再运行同一个分析器，并在分析 JSON 中记录 `source_decode_backend`；临时音频随后删除。

每个样本目录包含：

```text
<chart-id>/
├── audio.mp3       原始真实音频的硬链接、符号链接或本地副本
├── beatforge.json  生产版 BeatForge 分析与候选事件
├── chart.json      解析后的真实 SM 目标
└── metadata.json   来源、难度、哈希、split 与 realData 标记
```

数据集根目录还包含 `manifest.json`、`build_report.json` 与 `chart_statistics.json`。

## 本地 Transformer

安装可选 PyTorch 依赖后训练：

```bash
.venv/bin/pip install -e 'apps/api[chart-ml]'

.venv/bin/python scripts/train_chart_model.py \
  --dataset storage/chart-engine/dataset \
  --output storage/chart-engine/models/chart-transformer.pt \
  --epochs 12 \
  --batch-size 8 \
  --sequence-length 512 \
  --device auto
```

模型是带难度条件的 Transformer encoder。输入为当前歌曲真实 BeatForge candidate 序列，输出
五个独立面板概率和 Hold 概率。Checkpoint 记录固定特征 schema、归一化参数、数据集指纹、
真实样本来源、训练/验证损失、Torch 版本和设备，不包含音频本体。

当 `chart-transformer.pt` 存在时，生成接口默认使用它；checkpoint 不存在时会明确回退到基于
本地参考语料统计的确定性规则生成器。请求传入 `useLocalModel: false` 可主动关闭模型。损坏或
不兼容的 checkpoint 会返回 `LOCAL_CHART_MODEL_FAILED`，不会静默伪装成模型结果。

## 生成、优化与验证

生成器把 BeatForge `accepted` candidate 与持久化 hitPoint 当作节奏骨架；模型的五轨概率只参与
选键和 Hold，不再被误当成“事件是否存在”的概率。可选的 uncertain/rejected candidate 才会接受
模型阈值和难度预算筛选。同一量化槽只生成一个时序事件，同时在 `sourceEventIds`、
`sourceHitPointIds` 中保留合并前的完整来源，`sourceEventId` 继续作为兼容主 ID。

节奏网格随难度分级：Lv.1–3 为 1/4，Lv.4–7 为 1/8，Lv.8–10 最高为 1/16；只有 Lv.11–15
能根据低网格置信度的真实声学位置混合 1/16 与 1/24。连续 1/16 密集段保留为高压单键流，
随机或模型双押必须同时满足前后间距，不能用一个双押冒充两个参考标记。`enableSpin` 是独立开关，
默认关闭，开启后可插入三键小圈与五键大圈。

确定性优化器在密度超限时先把可选双押降为单键、再删除低置信可选事件；accepted marker 与
hitPoint 的时序行不会被静默删除。优化器和验证器共享按真实 BPM 校准的两秒整数 note 容量，
连续 1/16 可以通过，双押和持续 1/24 仍按每个 note 单独计数。验证器还会复核难度允许的实际
细分、同脚 16 分连续、身体位移、多键同时踩踏和 Hold 生命周期，并返回 0–100 分与逐项问题。

## API

本地授权语料：

- `GET /api/chart-engine/reference-charts`：筛选五轨/十轨、分组与标题。
- `GET /api/chart-engine/reference-charts/{chartId}`：读取完整绝对时间谱面。
- `GET|HEAD /api/chart-engine/reference-charts/{chartId}/audio`：支持 HTTP Range 的真实音频。
- `GET /api/chart-engine/statistics`：BPM、NPS、动作、脚法、转圈与轨道转移统计。

项目歌曲：

- `POST /api/tracks/{trackId}/chart/generate`：生成并验证谱面。
- `GET /api/tracks/{trackId}/chart/latest`：读取最近生成结果。
- `GET /api/tracks/{trackId}/chart/export?generationId=...`：导出 UTF-8 BOM `.sm`。

生成请求示例：

```json
{
  "difficulty": 10,
  "enableSpin": false,
  "useLocalModel": true,
  "seed": 20260721
}
```

响应中的 `chart.generator` 明确区分 `local_chart_transformer` 与规则生成器，
`chart.modelProvenance` 记录模型数据集指纹、checkpoint SHA-256 与 `realDataOnly` 标记；
`chart.optimization` 同时记录 accepted/hitPoint 锚点的输入输出行数，因此不同权重不会覆盖同一
生成版本，参考点覆盖也可逐次审计。

## 前端工作区

启动开发环境后：

- `/chart-engine` 浏览本地五轨参考谱面、语料统计与同步播放器。
- `/projects/{projectId}/chart` 为项目歌曲生成 1–15 难度谱面、切换转圈、查看验证结果并导出 SM。

Canvas 使用绝对 `timeSec` 绘制五条面板，音符自下向上接近判定线；HTML Audio 是唯一时钟，
播放、暂停、前后跳转和拖动进度会保持音频与谱面一致。

## 本地存储

```text
storage/chart-engine/
├── dataset/                 真实训练三元组、缓存与清单
├── models/
│   └── chart-transformer.pt 本地模型 checkpoint
└── generated/
    └── <track-id>/          版本化 JSON 与 latest.json
```

这些文件可由真实素材重新构建，默认不进入版本控制。分享训练产物或导出的 SM 前，请确认拥有
相应音频与谱面的使用权。

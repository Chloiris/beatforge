# BeatForge Studio · 卡点工坊

[![CI](https://github.com/Chloiris/beatforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Chloiris/beatforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-4FA89A.svg)](LICENSE)

BeatForge Studio 是一个本地优先的 AI 辅助音乐创作与音游制谱工作站。它把音频分析、
Demucs 分轨、日语 HuBERT CTC 歌词对齐、候选事件和可撤销时间轴编辑整合在同一个深色
DAW 风格界面中。

项目不会把音频或歌词发送到云端，也不需要 API Key。上传、模型、SQLite、分轨和分析结果
默认只保存在本机 `storage/`；这些内容均被 Git 和 Docker 构建上下文排除。

> 当前 v0.7.1 是可在 Windows、macOS 和 Linux 上运行的源码 / Docker 版本，不是签名的
> `.exe` 或 `.dmg` 安装包。基础编辑与分析由三平台 CI 验证；大型可选模型需要单独安装。

## 主要能力

- WAV、FLAC、MP3、M4A、AAC、OGG 导入与安全媒体读取。
- 多尺度、多频段、HPSS 增强的本地瞬态分析，以及 BPM、Offset 和细分网格。
- 可选 `htdemucs` 精确模式，生成 Mix、Vocals、Drums、Bass、Other 五条对齐波形。
- Japanese HuBERT CTC：保留观测 phoneme span，并映射为 Mora 与 Character 层级。
- Candidate Layer、Focus 段落、五轨独立人工标记和分轨试听。
- Canvas 时间轴缩放、定位、拖动、多选、锁定、吸附、Undo/Redo 与自动保存。
- 区分真实声学位置 `acousticSample` 与音游参考位置 `chartSample`，BPM 不覆盖声学证据。
- 一键导出含参考音频的 BeatForge 制谱包，兼容 JSON、CSV API。

## 导出数据包

编辑器顶部只有一个“导出”动作，直接生成供后续制谱工具使用的标准 `.beatforge.zip`。
数据结构、五轨拆分、双时间和参考音频均为固定内容，不需要在导出前选择套餐。

ZIP 中包含 `manifest.json`、候选事件，以及分别位于 `markers/` 下的
`mix.json`、`vocals.json`、`drums.json`、`bass.json`、`other.json`。每个标记同时记录真实
声音时间、制谱参考时间、偏移量、来源、置信度和证据，并附带采样时间轴一致的
`audio/reference.flac`。JSON、CSV、纯数据包与完整分轨包接口仍保留给自动化工具调用。完整格式说明见
[架构文档](docs/ARCHITECTURE.md)。

## 最快启动：Docker Desktop

这是 Windows 与 macOS 上环境差异最小的基础运行方式：

```bash
git clone https://github.com/Chloiris/beatforge.git
cd beatforge
docker compose up --build
```

打开 <http://127.0.0.1:5173>。API 文档位于 <http://127.0.0.1:8000/docs>。

Compose 只把端口绑定到本机回环地址，并把 `storage/` 挂载为本地数据目录。Docker 镜像只包含
基础分析，不包含 Demucs、HuBERT、Qwen 权重或 GPU/MPS 加速；完整模型功能建议使用原生安装。

## 原生安装

### 系统要求

- Python 3.11 或更高版本；CI 使用 Python 3.12。
- Node.js 20 或更高版本；CI 使用 Node.js 22。
- pnpm 11.9.0，可通过 Node Corepack 启用。
- FFmpeg 与 ffprobe；WAV/FLAC 仍可由 libsndfile 直接读取。
- 建议至少 8 GB 内存；Demucs 与人声模型建议 16 GB 以上。

### Windows 10/11 x64

先安装 Python、Node.js 和 FFmpeg，并确保 `python`、`node`、`corepack`、`ffmpeg`、`ffprobe`
可以在 PowerShell 中执行：

```powershell
corepack enable
python scripts/beatforge.py install
python scripts/beatforge.py seed
python scripts/beatforge.py dev
```

若 Windows 使用 Python Launcher，可把上述 `python` 换成 `py -3.12`。任务运行器会自动使用
`.venv\Scripts\python.exe`，不需要手工激活虚拟环境。

### macOS

使用 Homebrew 或其他系统包管理器安装 Python、Node.js、pnpm、FFmpeg 与 libsndfile，然后运行：

```bash
corepack enable
python3 scripts/beatforge.py install
python3 scripts/beatforge.py seed
python3 scripts/beatforge.py dev
```

打开 <http://127.0.0.1:5173>。API 默认位于 <http://127.0.0.1:8000>。

`seed` 是幂等的：它用固定随机种子生成三首合成演示音频、封面和 ground truth，再通过正常
分析路径写入本地数据库。合成 WAV 是可重建产物，不进入 Git。

## 可选模型

### Demucs 精确分轨

```bash
python scripts/beatforge.py install-accurate
python scripts/beatforge.py prepare-model
```

准备命令会显式下载 `htdemucs` checkpoint；正常分析不会隐式联网。macOS Apple Silicon 可用
MPS，Windows NVIDIA 可由 Demucs 使用 CUDA；Windows 无兼容 GPU 时回退 CPU。macOS 的可选
Demucs、HuBERT 与 Qwen 环境目前只验证 Apple Silicon，Intel Mac 不在可选模型支持范围内。

### Japanese HuBERT CTC / Qwen 草稿

```bash
python scripts/beatforge.py install-vocal
python scripts/beatforge.py prepare-alignment-models
```

如需 Qwen ASR 草稿与历史对齐器，再运行：

```bash
python scripts/beatforge.py prepare-vocal-models
```

模型安装在隔离的 `.venv-qwen` 与 `storage/models/` 中，不进入 Git。HuBERT/Qwen 在 Windows
当前使用 CPU；macOS Apple Silicon 可使用 MPS。Windows 安装 `pyopenjtalk` 时可能需要 C++
Build Tools 与 CMake。可选模型未进入日常三平台 CI，因此请在目标机器上单独验证后再用于生产。

## 开发与质量检查

跨平台任务运行器是主入口；macOS/Linux 也保留 Makefile 作为薄封装：

```bash
python scripts/beatforge.py doctor
python scripts/beatforge.py test
python scripts/beatforge.py lint
python scripts/beatforge.py build
python scripts/beatforge.py clean-generated
```

基础门禁包括后端 pytest、Ruff、前端 Vitest、TypeScript、ESLint 和 Vite 生产构建。GitHub
Actions 在 Ubuntu、macOS 与 Windows 上运行同一套命令，且不会下载模型。`test` 会在缺失时先
生成不入库的版权安全演示 WAV，因此干净克隆无需手动准备测试音频。

合成演示的确定性评估快照位于
[`reports/demo-evaluation.json`](reports/demo-evaluation.json)。它只验证仓库生成的合成音频，
不代表任意商业歌曲上的通用准确率。

## 隐私与安全

- 不要提交 `storage/` 中的上传、歌词、数据库、分轨、波形、模型或对齐产物。
- 不要提交私有项目截图、机器绝对路径或由用户歌曲生成的实验报告。
- API 没有身份认证或 TLS，只应绑定 `127.0.0.1`，不要直接暴露到局域网或公网。
- 制谱包默认包含参考音频；分享前请确认你拥有相应权利。
- 模型准备命令会访问对应上游仓库；准备完成后的分析设计为本地运行。

更多信息见 [SECURITY.md](SECURITY.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 数据目录

```text
storage/
├── audio/            用户上传
├── demo/             可重建的合成演示与 ground truth
├── covers/           合成 SVG 封面
├── stems/            Demucs 分轨
├── models/           可选模型权重
├── alignment/        Alignment Lab 结果
├── vocal-alignment/  模型子进程临时目录
├── waveform/         多级波形缓存
├── analyses/         分析快照
└── beatforge.db      SQLite 项目与编辑数据
```

`clean-generated` 只清理可重新生成的缓存，不删除用户原始音频。

## 项目结构

```text
apps/web/           React、Vite、Canvas 编辑器、Vitest、Playwright
apps/api/           FastAPI、SQLAlchemy、音频分析与 pytest
packages/shared/    跨端领域类型参考
scripts/            跨平台任务、模型准备、seed 与评估脚本
storage/            本地数据；仅合成 JSON/SVG 可进入 Git
reports/            仅允许合成演示评估进入 Git
docs/               架构、算法与实现说明
```

## 已知边界

- `other` 是 Demucs 的剩余声源，并不等同于钢琴识别。
- Focus 是声源活动的 soft routing 证据，不等同于歌词发音点。
- ASR 草稿需要人工校对；项目不会自动抓取歌词。
- 当前仓库不发布签名桌面安装器，也不捆绑 FFmpeg、模型或商业音频。

## License

代码使用 [MIT License](LICENSE)。依赖与模型的许可证说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

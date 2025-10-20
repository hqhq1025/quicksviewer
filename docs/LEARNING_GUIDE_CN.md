# Quicksviewer 代码学习指南

## 目录与源码职责
### 仓库顶层
- `README.md`：总览模型能力、安装方式、训练阶段脚本以及推理示例，是快速了解项目的入口。
- `AGENTS.md`：英文版贡献者指南，说明项目结构、格式化规范与提交要求。
- `Dockerfile`：用于构建包含训练依赖的 GPU 容器，脚本中预装 `torch`、`deepspeed` 与评测所需工具。
- `requirements.txt` / `pyproject.toml`：列出核心与可选依赖，同时在 `tool.black` 中设置 240 列宽的格式化规则。
- `docs/`：对外文档与补充资料，目前主要由 `README` 链接的网页内容组成。
- `playground/`：示例图片与视频资源，便于本地调试多模态推理。
- `scripts/`：训练与分布式配置。`quicksviewer/stage{1,2,3}.sh` 对应三阶段训练流程，根目录下 `zero*.json` 为 DeepSpeed 配置模板。
- `quicksviewer/`：Python 包主体，包含模型、训练、推理、预处理与评测代码。

### 包根目录
- `__init__.py`：暴露包级命名空间。
- `constants.py`：统一定义特殊 token（图像、视频、缩略图等）和忽略标签。
- `conversation.py`：实现对话模板与 prompt 生成逻辑，支持 LLaMA-3、Qwen2 等多种格式。
- `data/`：`dataset.py` 构建 `LazySupervisedDataset`、`PackedDataset` 等数据集，以及序列并行抽样；`preprocess.py` 负责将对话样本填充为训练输入。
- `eval/`：评测脚本，对应 `ActivityNet-QA`、`Video-MME`、`MVBench` 等数据集。`run_eval.sh` 聚合不同子命令。
- `model/`：模型组件。`builder.py` 负责加载权重与分配设备；`llava_arch.py` 定义视觉分支与“立方体”压缩逻辑；`language_model/` 下封装 LLaVA 定制的 LLaMA 与 Qwen2；`multimodal_encoder/`、`multimodal_resampler/`、`multimodal_projector/`、`multimodal_cubing/` 依次提供视觉塔、重采样、投影层和视频压缩策略；`make_delta.py`、`apply_delta.py` 用于参数增量导出与合并。
- `preprocess/`：离线数据清洗脚本，`convert_*.py` 针对不同公开数据（OBELICS、FineVideo、MovieChat 等）生成统一的 JSON/TSV；`run_prepro.sh` 给出批处理示例；`utils.py` 提供通用工具。
- `serve/`：`cli.py` 是命令行推理入口，`examples/` 存放体验用图片资源。
- `train/`：`train.py`、`train_hybrid.py`、`train_repack.py` 提供不同训练策略；`llava_trainer.py` 基于 HuggingFace `Trainer` 扩展序列并行和 Gumbel 温度退火；`args.py` 定义模型/数据/训练参数；`sequence_parallel/` 实现自定义张量通信；`utils.py` 包含学习率调度等工具。
- `utils/`：`data_util.py` 负责视频帧抽取、TSV 读取等 IO；`mm_utils.py` 处理高分辨率图像切分、停止准则；`utils.py` 封装日志与 Torch 初始化禁用。

## 建议的学习顺序
1. **阅读总览**：先通读 `README.md` 与 `AGENTS.md`，掌握项目目标、依赖环境以及提供的训练/评测命令。
2. **理解交互协议**：研读 `constants.py` 和 `conversation.py`，弄清特殊 token 与多模态对话模板的作用。
3. **掌握数据流**：按顺序查看 `quicksviewer/utils/data_util.py` → `quicksviewer/data/preprocess.py` → `quicksviewer/data/dataset.py`，了解样本如何从 JSON/TSV 转为模型输入。
4. **拆解模型结构**：从 `quicksviewer/model/builder.py` 入手，延伸阅读 `language_model/`、`multimodal_encoder/`、`multimodal_resampler/`、`multimodal_cubing/`，理解视觉编码、动态立方体压缩与投影流程。
5. **深入训练管线**：先读 `quicksviewer/train/args.py` 和 `train.py`，再看 `llava_trainer.py` 与 `sequence_parallel/`，掌握分布式配置、差异化学习率与自定义训练循环。
6. **补充工具链**：根据需要查阅 `preprocess/` 下的转换脚本、`serve/cli.py` 的推理逻辑以及 `eval/` 中的评测实现。
7. **实战演练**：参考 `scripts/quicksviewer/stage*.sh` 在小规模数据上跑通训练，再尝试 `serve/cli.py` 推理和 `eval/run_eval.sh` 评测，加深对数据与模型接口的理解。

## 模型运行流程
### 训练阶段
1. 调用入口通常为 `scripts/quicksviewer/stage{1,2,3}.sh`：脚本设置 `PYTHONPATH`、拼接数据集列表、选择 DeepSpeed `zero*.json`，最终通过 `python -m torch.distributed.run quicksviewer/train/train.py ...` 启动多机多卡训练。
2. `train.py::train()` 解析 `ModelArguments`、`DataArguments`、`TrainingArguments`，设置随机种子和序列并行拓扑（`sequence_parallel.set_pg_manager`）。
3. 加载基础语言模型 (`LlavaQwenForCausalLM` 或 `LlavaLlamaForCausalLM`)，按配置冻结/解冻骨干；调用 `model.initialize_vision_modules` 构建视觉塔、重采样器、Cubing 模块，并根据 `mm_*` 开关决定训练哪些参数。
4. 通过 `AutoTokenizer` 获取分词器，补充 PAD token 后写回模型配置，并在 `conversation_lib` 中设定默认模版。
5. 数据模块由 `make_supervised_data_module` 生成：`LazySupervisedDataset` 读取 JSONL/TSV，调用 `TSVReader` 抽帧，使用 `preprocess_multimodal_image/video` 注入特殊 token；`DataCollatorForSupervisedDataset` 负责打包、掩码 `IGNORE_INDEX`。
6. `LLaVATrainer` 扩展了 `_prepare_inputs` 以注入 Gumbel 温度 (`lr_gumbel`)；`get_train_dataloader` 使用长度分组采样保持多模态与纯文本的混合。
7. 训练过程在 HuggingFace `Trainer` 框架中进行，梯度支持 DeepSpeed ZeRO/LoRA/梯度检查点；每到保存步由 `safe_save_model_for_hf_trainer` 或 LoRA 分支落盘权重与配置。

### 推理与评测
1. `serve/cli.py` 解析命令行后调用 `utils.utils.disable_torch_init`，再通过 `model/builder.load_pretrained_model` 装载权重、构建视觉塔（必要时拆分到多 GPU）。
2. `parse_dialogue` 将交互式输入转为多回合消息，`load_image`/`load_video` 读取媒体格式；`mm_utils.tokenizer_image_token` 在 `model.chat` 前注入 `<image>` 占位及时间戳。
3. 模型生成过程依赖 `conversation_lib` 模板与 `TextStreamer` 实时输出，必要时使用 `KeywordsStoppingCriteria` 控制停止条件。
4. 评测脚本（如 `eval_video_mme.py`）调用统一的 `run_eval.sh`，内部加载和推理逻辑与 CLI 类似，但额外封装数据加载、指标聚合与结果持久化。

### 数据预处理
1. 若需要扩展数据集，先使用 `preprocess/convert_*.py` 将原始标注转为项目统一的 JSON/TSV；脚本依赖 `preprocess/utils.py` 的路径、抽样与压缩工具。
2. 最终生成的文件路径写入训练脚本的 `datasets` 字符串（由 `;` 分隔），与 `--video_folder` 指向的帧存储目录一起供 `LazySupervisedDataset` 消费。

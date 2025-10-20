# Quicksviewer 模型接入 SGLang 指南

本指南面向第一次对接 SGLang 的同学，详细讲解如何把本仓库的 Quicksviewer 视频多模态模型（Qwen2 语言骨干 + 动态 Cubing 压缩）接入到 `/Users/haoqing/Documents/Github/quicksviewer/sglang` 框架中。内容涵盖基础概念、准备工作、适配步骤、测试与排障建议。

---

## 1. 前置知识

### 1.1 SGLang 架构速览
- **核心目录**：`sglang/python/sglang/srt/`（Serving Runtime，简称 SRT）包含所有推理逻辑；`sgl-router/` 管理分布式调度；`sgl-kernel/` 封装 C++/CUDA 内核；`docs/`、`examples/` 提供文档与范例。
- **模型接入点**：所有可服务模型都在 `python/sglang/srt/models/` 下以单文件形式注册，通过 `ModelRegistry` 自动发现。
- **多模态处理链**：
  1. `model_config.py` 判断模型是否支持多模态并读取 HuggingFace 配置；
  2. `multimodal/processors/` 中的 `BaseMultimodalProcessor` 子类负责解析图片/视频输入，返回像素张量、哈希占位等结构；
  3. 模型类中的 `pad_input_ids`、`encode_images/get_image_feature` 等方法将多模态特征注入到语言模型。
- **对话模板**：`python/sglang/srt/parser/conversation.py` 维护 prompt 模板，服务端根据模板生成最终输入。

### 1.2 Quicksviewer 模型要点
- 位于 `quicksviewer/model/`，基于 LLaVA 架构扩展：
  - `language_model/llava_qwen.py`：自定义 `LlavaQwenForCausalLM`，整合多模态准备逻辑；
  - `llava_arch.py`：实现视觉塔、Cubing 压缩、Resampler 等组件；
  - `multimodal_cubing/`、`multimodal_resampler/`：核心创新，负责视频分块与重采样；
  - `builder.py`：推理阶段的权重加载入口。
- 训练/推理时依赖特殊 token（`DEFAULT_IM_START_TOKEN` 等）和多模态占位符，需要在 SGLang 中保留同样语义。

### 1.3 适配思路概览
SGLang 已内置多个视频/多模态模型（如 `qwen2_vl.py`、`llavavid.py`），Quicksviewer 的接入路径可以类比：
1. 让 config 的 `architectures` 与新模型类名一一对应；
2. 在 SGLang 中实现一个 PyTorch `nn.Module` 子类还原 HuggingFace 结构；
3. 实现匹配的 `MultimodalProcessor`，对接视频帧抽取与 token 替换；
4. 注册模型、处理器、对话模板并补充测试。

---

## 2. 适配准备

1. **保证代码可见**：确保执行环境的 `PYTHONPATH` 同时能访问 `quicksviewer/` 与 `sglang/`，建议在仓库根目录下执行相关脚本。
2. **导出 HuggingFace 权重**：
   ```bash
   # 示例：将 stage3 checkpoint 合并到 Qwen2 基座
   python quicksviewer/model/apply_delta.py \
     --base-model-path /path/to/Qwen2.5-7B-Instruct \
     --target-model-path /path/to/quicksviewer-s3-merged \
     --delta-path checkpoints/quicksviewer-s3/checkpoint-10000
   ```
   导出后检查生成目录内的 `config.json`，确认 `architectures` 字段（例如 `["LlavaQwenForCausalLM"]`）与模型类匹配。
3. **熟悉参考实现**：重点阅读 `sglang/python/sglang/srt/models/qwen2_vl.py` 与 `sglang/python/sglang/srt/models/llavavid.py`，理解 SGLang 如何处理视觉塔、RadixAttention 占位和连续批推理。

---

## 3. 适配步骤总览

1. **复制/改写模型结构**：在 `sglang/python/sglang/srt/models/` 下新增 `quicksviewer.py`（名称可自定义），继承合适的基类并引入 Quicksviewer 特有模块。
2. **实现多模态处理器**：在 `sglang/python/sglang/srt/multimodal/processors/` 下新增处理器，例如 `quicksviewer_processor.py`。
3. **注册模型与处理器**：
   - 在 `model_config.py` 中标记为多模态模型；
   - 确保 `ModelRegistry` 能发现新类；
   - 为 `BaseMultimodalProcessor.models` 添加新的 architecture 类。
4. **添加对话模板**：若默认模板不能满足视频 token 需求，在 `parser/conversation.py` 中注册新模板并指向新的 `conv_name`。
5. **打通 Server**：根据需要修改 `scripts/` 或自定义启动参数，确保 `--model` 指向新权重目录。
6. **测试验证**：与 HuggingFace 结果对齐、跑通 CLI 及服务端单元测试。

---

## 4. 详细步骤

### 4.1 复制模型结构

1. **新建文件**：`sglang/python/sglang/srt/models/quicksviewer.py`。
2. **引入依赖**：
   ```python
   import torch
   from transformers import Qwen2Config
   from quicksviewer.model.language_model.llava_qwen import LlavaQwenForCausalLM
   from quicksviewer.model.llava_arch import LlavaMetaModel
   from sglang.srt.layers.quantization.base_config import QuantizationConfig
   from sglang.srt.model_executor.forward_batch_info import ForwardBatch
   from sglang.srt.models.qwen2 import Qwen2Model
   from sglang.srt.managers.mm_utils import general_mm_embed_routine, MultiModalityDataPaddingPatternMultimodalTokens
   ```
3. **定义模型类骨架**（示例）：
   ```python
   class QuicksviewerForCausalLM(nn.Module):
       def __init__(
           self,
           config: Qwen2Config,
           quant_config: Optional[QuantizationConfig] = None,
           prefix: str = "",
       ) -> None:
           super().__init__()
           self.config = config
           self.language_model = Qwen2ForCausalLM(
               config,
               quant_config=quant_config,
               prefix=add_prefix("language_model", prefix),
           )
           self.vision_tower = build_quicksviewer_vision_tower(config, quant_config)
           self.resampler = build_quicksviewer_resampler(config, quant_config)
           self.cubing = build_quicksviewer_cubing(config, quant_config)
           self.multi_modal_projector = build_quicksviewer_projector(config, quant_config)
           self.image_feature_len = self.multi_modal_projector.out_features  # 自定义
   ```
   - 可以直接复用 `quicksviewer/model/multimodal_*` 中的实现，或把核心模块复制到 SGLang 模型文件内；
   - 注意在 SGLang 环境下禁止使用训练专用逻辑（梯度、loss），只保留推理路径。
4. **实现关键接口**：
   - `pad_input_ids(...)`：将 `<image>`、`<video>` token 替换为占位序列；参考 `llavavid.py` 或 `qwen2_vl.py`；
   - `get_image_feature(...)` / `encode_images(...)`：调用视觉塔与 Cubing 模块，生成语言模型接受的 embedding；
   - `forward(...)`：
     1. 获取文本 embedding；
     2. 若 `forward_batch.forward_mode.is_extend()`，根据 `mm_inputs` 把视觉特征插入 KV cache；
     3. 调用底层 `language_model.forward` 获取 logits，并执行 `LogitsProcessor`；
   - 需要兼容视频帧输入：处理 `forward_batch.mm_inputs.mm_items` 中的 `MultimodalDataItem`，按照帧顺序堆叠。
5. **权重加载**：SGLang 默认使用 `default_weight_loader` 自动加载权重；若 Quicksviewer 权重包含额外键名，需要实现 `from_pretrained` 或提供自定义 `weight_loader`。

### 4.2 实现多模态处理器

1. **新建文件**：`sglang/python/sglang/srt/multimodal/processors/quicksviewer.py`。
2. **继承 `BaseMultimodalProcessor`**，并标注支持的模型 architecture：
   ```python
   class QuicksviewerProcessor(BaseMultimodalProcessor):
       models = [QuicksviewerForCausalLM]
   ```
3. **核心方法**：
   - `_load_image_item` / `_load_video_item`：调用 `quicksviewer.utils.data_util.opencv_extract_frames_fps` 抽帧；
   - `pad_input_ids(...)`：与模型类对齐，返回替换后的 token 序列与 `MultimodalInputs`；
   - `preprocess_multimodal_prompt(...)`：照搬 `quicksviewer/data/preprocess.py` 中的逻辑，把 `<image>`、`<video>`、缩略图 token 打包进 prompt。
4. **注册处理器**：`BaseMultimodalProcessor` 会在首次导入时自动注册，只需保证 `models` 属性引用的类名与 HuggingFace config 的 `architectures` 对应。

### 4.3 配置对话模板

1. 如果沿用 Qwen2 的模板，可复用 `sglang/python/sglang/srt/parser/conversation.py` 中的 `register_conv_template("qwen2")`；
2. 若需要视频提示语或自定义停止词，新增模板：
   ```python
   register_conv_template(
       Conversation(
           name="quicksviewer",
           system_message="You are Quicksviewer...",
           roles=("HUMAN", "ASSISTANT"),
           sep_style=SeparatorStyle.LLAMA3,
           image_token="<image>",
           video_token="<video>",
           stop_str="<|im_end|>",
       )
   )
   ```
3. 启动服务时通过 `--chat-template quicksviewer` 指定；或在模型配置中设置默认模板。

### 4.4 修改模型注册逻辑

1. **标记为多模态**：在 `sglang/python/sglang/srt/configs/model_config.py` 的 `multimodal_model_archs` 列表中追加 `QuicksviewerForCausalLM`。
2. **确认自动注册**：`ModelRegistry` 会遍历 `sglang.srt.models` 包，只要新文件位于该目录且定义了对应类，即可自动收录。
3. **处理器导入**：在 `sglang/python/sglang/srt/managers/multimodal_processor.py` 中调用的 `import_processors("sglang.srt.multimodal.processors")` 会自动发现新处理器，无需手动修改。

### 4.5 打通运行路径

1. **本地离线测试**：
   ```bash
   cd /Users/haoqing/Documents/Github/quicksviewer/sglang
   python -m sglang.bench_one_batch \
     --model /path/to/quicksviewer-s3-merged \
     --correct \
     --input-images /path/to/sample.mp4 \
     --input-prompts "请详细描述视频内容"
   ```
2. **OpenAI 兼容服务**：
   ```bash
   python -m sglang.run_serving \
     --model /path/to/quicksviewer-s3-merged \
     --chat-template quicksviewer \
     --port 30000 \
     --tokenizer-mode slow  # 若使用自定义 tokenizer
   ```
3. 前端示例可参考 `sglang/examples`，替换成视频调用方式。

---

## 5. 测试与验证

1. **结果对齐**：使用 HuggingFace 推理脚本（`quicksviewer/serve/cli.py`）与 SGLang 输出对比，确保文本/评分一致；
2. **单元测试**：将模型名称加入 `sglang/test/srt/models/test_generation_models.py` 的 `ALL_OTHER_MODELS`，运行：
   ```bash
   ONLY_RUN=YourModelName python -m unittest test.srt.models.test_generation_models.TestGenerationModels.test_others
   ```
3. **性能评估**：参考 `sglang/docs/developer_guide/benchmark_and_profiling.md`，在目标 GPU 上跑 TTFT / Tokens per Second，对比基线模型（如 Qwen2-VL）。
4. **多模态评测**：可对接 `quicksviewer/eval/run_eval.sh` 的数据集（Video-MME、MMMU 等），验证准确率未下降。

---

## 6. 常见问题与排障

| 问题 | 排查思路 |
| ---- | -------- |
| 模型加载时报 `KeyError` | 检查 `config.json` 的 `architectures` 是否更新；确认新类名已在 SGLang 模型目录中定义；必要时 override `state_dict` 键映射。 |
| 服务端无法识别 `<video>` token | 确保 `pad_input_ids` 和 `MultimodalProcessor` 输出的占位长度一致；查看 `forward_batch.mm_inputs` 中的 `image_offsets`。 |
| 视觉特征形状不匹配 | 仔细核对 `quicksviewer` 中 `resampler`、`cubing` 输出维度，与 SGLang 模型类的 `image_feature_len` 设置保持一致。 |
| 多卡推理报错 | 检查 `vision_tower` 设备分配，必要时在 `forward` 中调用 `to(device=...)`；确认 `QuantizationConfig` 兼容。 |
| 推理结果与 HF 相差大 | 逐步打印 `logits` 或 `hidden_states`；使用 `python -m sglang.bench_one_batch --correct` 对比两边 `prefill` logits。 |

---

## 7. 后续优化建议

1. **封装公共代码**：将 Quicksviewer 的视觉压缩模块封装为独立包，避免在 SGLang 中大量复制代码。
2. **补充样例脚本**：在 `sglang/examples` 新增视频问答 Notebook，演示如何调用 Quicksviewer。
3. **量化与加速**：探索 SGLang 提供的 INT4、FP8、RadixAttention Chunked Prefill 等功能，评估性能收益。
4. **CI 回归**：若打算贡献给社区，可在 `test/srt/test_vision_openai_server_*.py` 中新增对应测试，并在 PR 中附上基准分数。

---

完成以上步骤后，即可在 SGLang 上稳定运行 Quicksviewer，多模态视频问答、描述与分析任务均可通过统一的 API 对外提供服务。建议在适配过程中持续记录配置修改与实验结果，便于后续维护与团队协作。祝开发顺利！ 🎉

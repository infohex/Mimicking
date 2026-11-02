[English](license-review.md) | 中文

# 许可审读（2026-10-27）

> 2026-11-01 更新：2.0 正式版的工具链已不含任何 GPL 代码（XLS-R / CAM++ / HiFT 由 MimickingTrain 从 Apache-2.0 权重直接导出），下文提到的 `tools/export`、`ref/seed-vc`、`bench/` 已归档到 `C:/Workspace/Rust/_archive/Mimicking-seedvc-2026-11-01`，只在需要重导基线套或做基线对拍时取用。下文其余分析为 2026-10-27 的原文。
>
> 2026-11-01 决定：库代码采用 Apache-2.0（已加 `LICENSE` 与 `NOTICE`，Cargo `license = "Apache-2.0"`）。下文第 2 条因此已定；其中讨论的 GPL 相关材料已归档、不分发。


逐项读过的协议：上游 seed-vc 仓库 LICENSE、Hugging Face 四个模型卡、CosyVoice 与 3D-Speaker 仓库 LICENSE、
ModelScope CAM++ 模型页、RVC 仓库 LICENSE、NVIDIA CUDA EULA 与 cuDNN SLA、onnxruntime 与全部 Rust 依赖的许可字段。
这是工程侧的整理，不是法律意见；对外商用前仍需法务确认，尤其是第 1 条。

## 一览

| 组件 | 我们怎么用 | 许可 | 出处 | 结论 |
|---|---|---|---|---|
| seed-vc 代码（Plachtaa/seed-vc，2025-11-21 已归档） | `ref/` 只读参考；`tools/export` 的 Python 脚本 **import 它的模块**来导出图 | **GPL-3.0**（仓库 LICENSE） | github.com/Plachtaa/seed-vc | Rust 库不含其代码；但 `tools/export` 链接了 GPL 代码，若对外发布这些脚本须按 GPL-3.0 |
| DiT 流匹配估计器 + 长度调节器权重 `DiT_uvit_tat_xlsr_ema.pth` | 导出成 `dit_step*.onnx`、`length_regulator*.onnx` | **GPL-3.0**（HF 模型卡仅有 license 标签，无正文、无训练数据说明、无商用说明） | huggingface.co/Plachta/Seed-VC | **主要风险点**，见第 1 条 |
| XLS-R `wav2vec2-xls-r-300m` | 导出成 `xlsr*.onnx` | Apache-2.0（HF 模型卡；训练数据 VoxPopuli、MLS、CommonVoice、BABEL、VoxLingua107，436k 小时） | huggingface.co/facebook/wav2vec2-xls-r-300m | 可商用、可再分发，附许可与声明即可 |
| HiFT 声码器 `hift.pt` | 导出成 `hift_f0.onnx`、`hift_body.onnx`、`hift_source.npz` | Apache-2.0（HF CosyVoice-300M 模型卡标签；CosyVoice 仓库 LICENSE 为 Apache-2.0；seed-vc 内 `modules/hifigan/*.py` 文件头亦标 Alibaba Apache-2.0） | huggingface.co/FunAudioLLM/CosyVoice-300M | 可商用。模型卡有"内容仅供学术展示"的免责声明，指的是示例音频，不是权重许可 |
| CAM++ `campplus_cn_common.bin` | 导出成 `campplus.onnx`；测试台质量探针 | Apache-2.0（HF funasr/campplus 标签；ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` 许可字段 Apache-2.0；3D-Speaker 仓库 Apache-2.0；seed-vc 内 `modules/campplus/*.py` 文件头标 3D-Speaker Apache-2.0） | ModelScope / FunASR | 可商用。训练数据为"约 20 万中文说话人的大型数据集"，来源未公开 |
| RVC（实时切块、交叉淡化、SOLA 的算法来源） | seed-vc 的实时逻辑"修改自 RVC"；我们的 Rust 是按算法重写 | MIT（liujing04） | github.com/RVC-Project | 无义务；建议在致谢中注明 |
| onnxruntime-gpu 1.22（3 个 DLL） | 随运行目录分发 | MIT | microsoft/onnxruntime | 附 LICENSE 即可 |
| CUDA 12 运行库（cudart、cublas、cublasLt、cufft） | 随运行目录分发（从 torch/lib 取出） | NVIDIA CUDA EULA | docs.nvidia.com/cuda/eula | 附件 A 明列可随应用分发；见第 3 条的条件 |
| cuDNN 9（cudnn*.dll） | 随运行目录分发 | NVIDIA cuDNN SLA | docs.nvidia.com/deeplearning/cudnn/sla | 运行时 .dll 可随应用分发；条件同上 |
| Rust 依赖（库 ~110 个 crate） | 静态链接进 `mimic.exe` | 全部宽松：MIT / Apache-2.0 / BSD / ISC / Zlib / Unlicense；`unicode-ident` 含 Unicode-3.0 声明；`r-efi` 三选一取 MIT | crates.io | 发布时附 NOTICE（`cargo about` 或 `cargo license` 生成） |
| 测试台依赖（363 个 crate，不对外分发） | 仅本地 | 同上，另有 BSL-1.0（Boost）、OFL-1.1 字体、`self_cell` 为 Apache-2.0 OR GPL-2.0（取 Apache） | — | 无需处理 |
| 我们自己的代码 | — | Apache-2.0（2026-11-01 定稿；`LICENSE`） | — | 见第 2 条 |

## 1. GPL-3.0 的权重：真正的问题

事实：

- seed-vc 代码仓库是 GPL-3.0；Hugging Face 上的权重页只有一个 `gpl-3.0` 标签，README 为空，没有说明训练数据、
  没有说明是否允许商用、没有任何补充条款。作者在 GitHub 和 HF 都没有区分"代码许可"和"权重许可"。
- 我们从这份权重导出的 `dit_step.onnx`、`dit_step_fused.onnx`、`length_regulator*.onnx` 是同一组参数的另一种存储形式
  （转置、fp16、融合算子，数值等价），不是独立创作。
- 仓库已归档（2025-11-21），作者不再维护，但许可不会因归档失效。

怎么理解：

- GPL 是为软件写的，对"权重是否构成受版权保护的作品"法律上尚无定论。但把标了 GPL-3.0 的文件拿来分发，最稳妥的
  假设是：**分发权重（含我们导出的图）就要遵守 GPL-3.0**——允许商用，但接收方有权再分发、再修改，且要随附许可文本
  和"源码"（对权重而言就是权重本身）。
- 我们的 Rust 程序本身不是权重的衍生作品：它是独立实现，运行时读入权重作为数据。GPL 的传染只及于"基于该作品的作品"，
  读数据的程序一般不算。因此 **程序可以保持 MIT/Apache，权重另算**。这是普遍的解读，但没有判例，法务可能更保守。
- 真正会被认定为 GPL 衍生作品的是 `tools/export` 下的 Python 脚本：它们 `import modules.flow_matching`、
  `import modules.diffusion_transformer` 等 GPL 模块，并以 `DiTClean` 重写了同一前向。这些脚本若对外发布，必须按 GPL-3.0。
  seed-vc 内 `modules/hifigan`、`modules/campplus` 两个目录的文件头是 Apache-2.0，不在此列。

可选路线：

1. **权重与程序分开分发**：程序目录 MIT/Apache，模型目录单独提供并附 GPL-3.0 文本与来源说明（manifest 已记录）。
   接收方对模型目录享有 GPL 权利。适合内部部署和不介意模型可再分发的客户。
2. **向作者要授权**：联系 Plachta 取得权重的单独商用许可或明确声明。仓库已归档，回复不确定。
3. **自训权重**：用 seed-vc 的训练代码（GPL）训练，**训练程序的输出不受 GPL 约束**（GPL 第 2 条：程序输出只有在
   本身包含程序内容时才受约束），得到的权重归我们。代价是数据与算力，并且要留意训练数据自身的许可。
4. **换掉 GPL 部件**：DiT 与长度调节器是 GPL 的唯一来源；XLS-R、HiFT、CAM++ 都是 Apache-2.0。换一个 Apache/MIT
   许可的流匹配声谱生成器即可整体宽松，但等于换模型，质量要重新评估。

另一个事实上的风险与许可无关：tiny 权重的训练数据未公开，无法判断是否含有未经授权的音色。对外产品需要把
"目标音色只来自客户授权的参考录音"写进使用条款。

## 2. 我们自己的代码

- 现状：2026-11-01 定为 Apache-2.0（`LICENSE`、`NOTICE`、Cargo 元数据）。下面两条是当时的分析。
- 若选 MIT/Apache：仓库里必须把 `tools/export` 单独标为 GPL-3.0（或不随库发布，只在内部用），README 写明。
- 若整个仓库 GPL-3.0：最省事也最一致（与 tools/export、权重同一许可），代价是任何集成方都要开源其整体。
- 无论哪种，发布包要附：我们的 LICENSE、onnxruntime 的 MIT 文本、XLS-R/HiFT/CAM++ 的 Apache-2.0 文本与 NOTICE、
  Rust 依赖的汇总 NOTICE、CUDA/cuDNN 的 EULA 指引、RVC 致谢。

## 3. NVIDIA 运行库的分发条件

CUDA EULA 附件 A 明列 cudart、cublas、cublasLt、cufft 的 DLL 可随应用分发；cuDNN SLA 允许分发运行时 .dll。共同条件：

- 只能作为**我们应用的一部分**分发，不能单独或当作 SDK 分发；应用要有实质功能（满足）。
- 只有我们的应用访问这些库（运行目录结构满足：`runtime/cuda` 只被 `mimic.exe` 加到 PATH）。
- 我们的分发条款不得与 EULA 冲突，须保护 NVIDIA 的知识产权；禁止逆向、去除版权声明；
  **"不得以会使 SDK 受开源许可约束的方式使用"**——这一条和第 1 条有交互：如果整个发布包按 GPL-3.0 分发，把 NVIDIA 的
  DLL 放在同一个 GPL 包里是有争议的。安全做法：运行目录（程序 + CUDA/cuDNN/ORT）按我们的宽松许可或专有条款分发，
  模型目录按 GPL 单独分发；或者不随包带 CUDA，要求目标机安装 CUDA 12 / cuDNN 9 运行库。
- 不得用于"关键应用"（航空、医疗、自动驾驶等）。
- 从 `torch/lib` 取 DLL 只是取得方式，这些文件的许可仍是 NVIDIA 的，不是 torch 的 BSD-3。

## 4. 其余部件：无障碍

- XLS-R、HiFT、CAM++ 三份 Apache-2.0 权重：可商用、可修改、可再分发；义务是附许可文本、保留 NOTICE、注明修改。
  我们导出的 ONNX 图属于"修改"，在 manifest 和 NOTICE 里写明即可。
- onnxruntime MIT、Rust 依赖全部宽松。
- RVC MIT：seed-vc 的实时窗口逻辑来自它，我们按算法重写；致谢即可。

## 5. 建议的动作

1. BO 决定第 1 条走哪条路（分开分发 / 要授权 / 自训 / 换模型），这是产品决策。
2. 在此之前：仓库 README 的许可一节写明三类（我们的代码、tools/export 为 GPL-3.0、模型按各自上游）；
   `make_manifest.py` 已把四个权重的许可都写进 manifest（本次补上 CAM++ 与 RVC 致谢）。
3. 打包脚本 `package_runtime.py` 以后生成 `THIRD-PARTY-NOTICES`：ORT MIT、三份 Apache-2.0、Rust 依赖汇总、NVIDIA EULA 指引。
4. 对外文档（导览页）中"模型许可为 GPL-3.0，需法务确认或改用自训权重"这一句保持不变。

## 6. 自训权重的可行性（2026-10-27 评估）

读了 `train.py` 后的事实：

- 训练只更新 `cfm`（DiT）和 `length_regulator`；XLS-R、CAM++、HiFT 冻结，全是 Apache-2.0。正是 GPL 的两个部件可以自训。
- 支持从零训练：配置里 `pretrained_model: ""` 且不传 `--pretrained-ckpt` 即随机初始化。**不能**从 GPL 检查点微调，那仍是衍生。
- 训练时用 OpenVoice（MIT，含 checkpoints_v2）做音色扰动增强，并从 Plachta/Seed-VC 下载 `se_db.pt`（说话人嵌入库，GPL 标签）
  当扰动目标。`se_db` 只是训练输入，不进权重；稳妥起见用 OpenVoice 的 `extract_se` 从自己的数据重建一份。
- GPL 训练程序的输出（权重）不受 GPL 约束（GPL-3.0 第 2 条）；架构本身不受版权保护；Rust 推理端和导出脚本无需改动。
- 数据要求：1–30 s 的干净多说话人语音，无需标签；数据的许可决定成品能否商用：
  LibriTTS-R（CC BY 4.0，585 h 英文）、MLS（CC BY 4.0，8 语种，英文数万小时）、AISHELL-3（Apache-2.0，85 h 中文 218 人）可商用；
  Emilia（CC BY-NC 4.0）、WenetSpeech4TTS（使用条款限非商业）不可。
- 算力：v1 `train.py` 单卡；25M 参数，batch 2 × 100 s，每步冻结 XLS-R 跑两遍（原始 + 扰动）是大头，A4000 上估计 0.5–1 s/步，
  从零训练按 30–100 万步计 3–10 天一张卡，16 GB 够用；两张卡可并行两组实验。
- 风险在质量而不在工程：公开 tiny 权重的训练数据未知，自训版的相似度与可懂度要用测试台重新评估；中文商用数据是瓶颈
  （AISHELL-3 太小，需另购或自采授权数据）。

# SpotEdit — 新 backbone 适配 + 测试总结 (2026-06)

把 SpotEdit（training-free 区域感知编辑加速）适配到两个新 backbone 并在 H200 上真权重验证。

## 1. 新增模块（都在本目录下，GitHub 风格，每个 backbone 5+ 文件）

### `Qwen_image_edit_plus/` — Qwen-Image-Edit **2509 / 2511**
同一个 `QwenImageEditPlusPipeline`，2509/2511 只换 repo id。
- `qwen_plus_spotedit.py` — `generate()`（多图预处理 384²+1024²、多条目 img_shapes）
- `qwenSpotAttn.py` — KV 缓存 attention 处理器（修了多图 `image_n≠latent_n2` bug）
- `QwenTokenLPIPS.py` — VAE token-LPIPS 复用判据
- `qwen_spot_ultis.py` — `SpotEditConfig` / `Spotselect` / dilation / feather / boundary-smoothing
- `examples/qwen_plus.ipynb`, `examples/qwen_plus_2511.ipynb`

**配置项（`SpotEditConfig`）**
- `reuse_mode`：**`"velocity"`（默认，最佳）** 复用 token 用 `v=(x_t−x0_orig)/σ` 直线流向原图 → 无接缝；`"feather"` 像素域内容感知羽化合成；`"overwrite"` GitHub 原版（latent 硬贴+边界平滑，有方块缝）
- `source_init_strength`：<1.0 = SDEdit 起点（从加噪原图开始，更保真+更快但编辑更保守）
- `select_every_step`：每步重算复用掩码 vs 每个 reset block 算一次（每步会让 LPIPS 开销吃掉加速）
- `judge_method`：`LPIPS`(默认) / `cosine` / `L4`；`threshold` / `dilation_radius` / `reset_steps` / `initial_steps`

### `FLUX2/` — FLUX.2
- `flux2_spotedit.py` — dev generate（`Flux2Pipeline`，Mistral-3 编码器，需 diffusers 0.36+）
- `flux2klein_spotedit.py` — **klein generate**（`Flux2KleinPipeline`，Qwen3 编码器，需 diffusers ≥0.37）
- `flux2SpotAttn.py` — **两个**处理器：双流 `Flux2SpotAttnProcessor` + 单流并行 `Flux2ParallelSpotAttnProcessor`
- `FLUX2LPIPS.py` — 适配 `AutoencoderKLFlux2` 的 batch-norm + 二次 patchify 的 token-LPIPS
- `flux2_spot_ultis.py` — config / SpotSelect / dilation
- klein 复用了 dev 的 attn 处理器 + LPIPS（共享 `Flux2Transformer2DModel` + `AutoencoderKLFlux2`）

通用修复：两个模块的 `generate()` 退出时都**恢复原始 attn 处理器**（否则之后再调 `pipe()` 会带着残留缓存崩）。

## 2. 真权重测试结果（H200）

| backbone | 加速 | 画质 | 备注 |
|---|---|---|---|
| Qwen-Image-Edit-2509 | ~2.2× | 干净(velocity) | 4 个种子(0/7/42/123)一致；红/蓝围巾都验过 |
| Qwen-Image-Edit-2511 | ~2.3× | 干净 | — |
| FLUX.2-klein-4B | **1.88×** | 干净蓝围巾 | klein 适配 + 隔离 diffusers 0.39；klein-4B 小所以加速略低 |
| FLUX.2-dev | 仅 CPU 验证 | — | gated + 177GB（被大小/授权挡住，非代码问题）|

**加速来源（per-stage 计时，40 步 reset8）**：transformer 占 base 时间 ~93%；缓存步只算 ~6% 的 token、比全量步快 ~4×（K/V 仍全长 + 每层重组开销封顶在 ~4×）；LPIPS 选区 ~136ms/步（所以 select-every-step 会掉到 1.6×）。**reset 频率是 速度↔画质 的主旋钮**（reset4→1.83× / reset8→2.29×）。对照实验证明加速真实：用同一份 `generate()` 关掉缓存 = 19.82s ≈ stock base 19.96s，开缓存 = 8.71s。

**CPU 正确性**：`test_spotedit_cpu.py` → **8/8**（no-cache 步与 stock 处理器逐位相同；缓存复用未变 token == 全量重算逐位相同；FLUX2 LPIPS 解包是 pipeline 打包的精确逆；多图 KV 重组正确）。

## 3. 环境

- 主 env：conda `spotedit` @ `/scratch/zhibin.qin/miniconda3/miniconda3/envs/spotedit`（diffusers **0.36.0**、torch 2.9.1+cu128、transformers 4.57.3）。有 `Flux2Pipeline` + `QwenImageEditPlusPipeline`。python = 该 env/bin/python。
- klein 专用：diffusers **0.39.0.dev0** 隔离装在 `/scratch/zhibin.qin/diffusers037`（`pip install --target --no-deps git+diffusers`，复用主 env 的 torch/transformers）。用法：`PYTHONPATH=/scratch/zhibin.qin/diffusers037 python ...`。**不动主 env**，0.36.0 的 Qwen 验证不受影响。
- HF token 存在 `/scratch/zhibin.qin/hf_cache/token`（你提供的，用于 gated 下载；不需要可删）。
- 已缓存模型（`/scratch/zhibin.qin/hf_cache`）：Qwen-Image-Edit-2509、2511、FLUX.2-klein-4B。

## 4. 怎么跑（在本目录根下；GPU 节点：`qsub -I -q interactive -P CFP04-CF-011 -l select=1:ngpus=1`，再 `export HF_HOME=/scratch/zhibin.qin/hf_cache`）

```bash
PY=/scratch/zhibin.qin/miniconda3/miniconda3/envs/spotedit/bin/python
# Qwen (2509/2511)：参数 = 步数 reset间隔 SDEdit强度 每步选区(0/1)
QWEN_REPO=Qwen/Qwen-Image-Edit-2511 PROMPT="add a red scarf..." $PY smoke_qwen_plus.py 40 8 1.0 0
# FLUX.2-klein-4B：需要隔离 diffusers
PYTHONPATH=/scratch/zhibin.qin/diffusers037 $PY smoke_flux2klein.py 50 8
# CPU 正确性（无需 GPU/权重）
$PY test_spotedit_cpu.py
# per-stage 计时对比 (base vs spotedit, 含关缓存对照)
$PY time_qwen.py 40 8
```

## 5. eval/ 子目录
- `eval/logs/` — 所有运行日志
- `eval/results/` — 所有对比图/结果图（`compare_2509.png`、`compare_blue_2509_vs_2511.png`、`seeds_2509.png`、`flux2klein_spotedit.png` 等）

## 6. 待办 / 局限
- FLUX.2-dev 真权重：需 HF token(已有) + 删两个 Qwen 腾 ~177GB + 数小时下载（建议批作业），用 `FLUX2/flux2_spotedit.py`。
- velocity 模式目前在 Qwen + klein 路径；`FLUX2/flux2_spotedit.py`（dev）还是 overwrite+boundary，建议也切 velocity。
- 这些改动尚未推 GitHub（用户要求暂不推）。

## 7. 参数消融 + overwrite 复检（2509，blue scarf，40 步，single-forward base 19.9s）

`ablation_2509.py`（base + 9 配置 montage → `eval/results/ablation_2509.png`）：

| 配置 | 时间 | 加速 | mean\|Δ\| | 结论 |
|---|---|---|---|---|
| overwrite (reset8,dil2) | 8.9s | 2.23× | 3.0 | **原版硬替换，2509 上无方块缝** |
| feather  (reset8,dil2)  | 8.9s | 2.24× | 3.0 | 干净 |
| velocity (reset8,dil2)  | 8.7s | 2.27× | 2.9 | 干净（默认）|
| velocity thr=0.10 | 8.7s | 2.27× | 2.9 | 阈值低→复用略少，本编辑影响很小 |
| velocity thr=0.25 | 8.7s | 2.28× | 3.3 | 阈值高→复用更多，略快+略漂 |
| velocity reset=4  | 10.7s | 1.86× | 2.7 | 密 reset：最慢最保真 |
| velocity reset=16 | 7.8s | 2.55× | 3.9 | 疏 reset：最快、漂移最大 |
| velocity dil=0    | 8.3s | 2.38× | 4.0 | 无膨胀：重算框小→快但边界漂 |
| velocity dil=4    | 9.0s | 2.20× | 3.0 | 大膨胀：重算框大→慢、边界略干净 |

**主旋钮 = reset 频率**（速度↔画质单调：reset4 1.86× / reset8 2.27× / reset16 2.55×，Δ 2.7→3.9）。膨胀次之（dil0 Δ4.0 → dil2 Δ2.9，膨胀保边界保真）。阈值与合并模式在 2509/blue-scarf 上影响都很小（三种合并模式 Δ≈3）。

**overwrite 接缝复检**（`seam_check_2509.py` → `eval/results/seam_check_2509.png`，全分辨率 base/overwrite/velocity + ×8 放大 \|Δ\| 热图）：overwrite vs base max=113/mean=2.97，velocity vs base max=112/mean=2.92，两张差异热图**几乎一致**，差异只是重算区（狗+围巾）的有机纹理漂移，**没有矩形缝**，背景近全黑。结论：**接缝是 config/编辑/backbone 相关的**（2511、或复用区在编辑边界带结构时才显现），不是普遍 artifact；2509 这个编辑里重算掩码贴合狗的自然轮廓，硬贴边界落在真实物体边上 → 不可见。velocity 仍是"永不更差"的安全默认。

## 8. 非编辑区保真度 + mask 可视化（多 seed × 2509/2511）★核心结果

`generate()` 新增可选 `aux: dict` 出参，回传最终复用掩码 `aux["reuse_mask"]`（latent grid，True=非编辑/复用）+ `H_lat/W_lat`，用于可视化"冻结了哪块、动了多少"。

**`seam_seeds.py`**（base/overwrite/velocity + ×8 \|Δ\|-vs-base，4 seed×2 model → `eval/results/seam_seeds_{2509,2511}.png`）：

| model | \|Δ\|-vs-base | overwrite vs velocity | 备注 |
|---|---|---|---|
| 2509 | ~3（4 seed 全一致）| 几乎相同 | 差异图背景全黑、只在狗/围巾 |
| 2511 | **~13-16** | velocity 矩形边界更平滑 | 差异图背景**发亮** |

2511 的 \|Δ\|-vs-base 大不是 SpotEdit 错，而是 **2511 的 single-forward base 把背景橙色 drift 得更饱和**，SpotEdit 复用原图 token → 留在原图色调 → 与 base 差很多。**说明"对 base 比"是错的参照系**，要"对原图比"。

**`nonedit_viz.py`**（ORIGINAL / spot / 非编辑 mask 绿色高透明叠加 / spot Δ-vs-**原图** ×8 / base Δ-vs-原图 ×8，3 seed×2 model → `eval/results/nonedit_viz_{2509,2511}.png`）：

| | 非编辑区占比 | **SPOT 非编辑 Δ-vs-原图** | BASE 非编辑 Δ-vs-原图 | 围巾(编辑区) Δ |
|---|---|---|---|---|
| 2509 (3 seed) | ~0.84 | **2.11–2.14** | 2.80–3.11 | 50–70 |
| 2511 (3 seed) | ~0.90 | **2.17–2.18** | **14.6–16.9** | 35–56 |

**核心结论**：SpotEdit 的非编辑区在**两个 backbone 上都只动 Δ≈2.1–2.2/255**（≈VAE round-trip 地板，即"动都不要动"已达成，且与 base 漂多少无关）；而 full-recompute base 会动——2509 上动 ~2.9，2511 上动 **~15**。可视化里 spot 的 Δ 热图除围巾外**全黑**（非编辑区没动），base 的 Δ 热图 2511 上整张狗+背景**发亮**（全图被重画）。所以 SpotEdit 的"区域冻结"在 base 漂得越狠的 backbone 上价值越大（2511 上比 base 保真 ~7×）。绿色 mask 叠加图直观展示被冻结区（绿）vs 唯一活跃的编辑区（围巾，不染色）。

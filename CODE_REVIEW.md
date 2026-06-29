# SpotEdit 代码评审 (2026-06-29)

> 评审方式:9 路 agent 分维度静态评审 + 对抗式验证(55 条经验证存活 → 去重为 33 条),
> 关键项逐条对照源码人工复核。范围:4 个 backbone(`FLUX_kontext`、`FLUX2`、
> `Qwen_image_edit`、`Qwen_image_edit_plus`)共 35 个 `.py` + 5 个 notebook。
> 可执行清单见 [`TODO.md`](./TODO.md)。

## 结论先行

**核心思路成立,最难的部分是对的。** KV-cache 机制——cache 重组、未缓存 query 的 RoPE
子选择、`_reassemble_kv`——在充分测试的路径(`FLUX2`、`Qwen_image_edit_plus`)上**正确**:
CPU 正确性测试(`test_spotedit_cpu.py`)和验证 agent 都确认它能逐近(~1e-6)复现全量重算
结果。那个看着吓人的"双重 `/1000`" KV 混合其实是**惰性空操作**,碰巧落在了正确的缓存值上,
不是真 bug。

**但整体还不能直接发布。** 几乎所有严重缺陷都在外围,而 `FLUX_kontext` 是最薄弱的一环——
它对外宣传的入口**一 import 就崩**。`FLUX2` / `qwen_plus` 两条路接近可发布;另外两个
backbone 需要返工。

**建议的发布闸门:** 先修掉 🔴严重 #1 + 🟠高 #2–#6,再考虑发布,其余可随后跟进。

---

## 🔴 严重(1)— 阻塞发布

### 1. `FLUX_kontext` 用了裸 import → 文档入口 import 即崩
- **位置:** `FLUX_kontext/flux_spotedit.py:15-16`、`flux_spot_ultis.py:15`、`flux_spotedit_tra.py:15-16`
- **问题:** 写成 `from flux_spot_ultis import ...` / `from fluxSpotAttn import ...` /
  `from FLUXLPIPS import ...`(无前导 `.`)。Python 3 无隐式相对导入,README 里的
  `from FLUX_kontext import ...` 会抛 `ModuleNotFoundError`。另外三个 backbone 均用 `from .xxx`。
- **修复:** 这五处改成相对导入。
- **状态:** 已直接确认(对照 `FLUX2/flux2_spotedit.py:28` 的 `from .flux2SpotAttn import`)。

---

## 🟠 高(5)— 常规路径上会出错 / 崩溃 / 泄漏

### 2. attention processor 用完不还原 → 跨调用污染 pipeline
- **位置:** `Qwen_image_edit/qwen_spotedit.py`、`FLUX_kontext/flux_spotedit.py`(均缺还原);
  `FLUX2/flux2_spotedit.py:313-315`、`Qwen_image_edit_plus/qwen_plus_spotedit.py:441-442`(有还原但无 `try/finally`)
- **问题:** `Qwen_image_edit` 和 `FLUX_kontext` 装上 SpotEdit processor(闭包持有本次
  `cache_flags`)后**从不还原**(`grep _orig_procs` = 0)。之后再调普通 `pipe()` 会带着残留
  缓存进入 cached 分支 → token 切错 / 形状崩 / 静默出错。`FLUX2`、`qwen_plus` 会还原,但
  **只在成功路径**(无 `try/finally`),循环里 OOM/异常会跳过还原。
- **修复:** 安装前快照 `_orig_procs`,把 安装+循环+收尾 包进 `try/…finally:` 还原;给缺失的两个
  backbone 补上完整快照/还原。

### 3. 默认 LPIPS 判据把 `image_size` 写死成 `(1024,1024)` → 非方图崩
- **位置:** `FLUX_kontext/flux_spot_ultis.py` 的 `SpotSelect`、`Qwen_image_edit/qwen_spot_ultis.py` 的 `Spotselect`
- **问题:** `judge_method` 默认 `'LPIPS'`,但这两个 backbone 的 LPIPS 分支传字面量
  `(1024,1024)`,忽略真实 `height/width`。两条 pipeline 保宽高比,非方图 token 数 ≠ 4096 →
  第一个 reset block 就 `ValueError`/`RuntimeError`。`Qwen_image_edit` 还写死 `self.vae.to('cuda')`。
  `FLUX2`/`qwen_plus` 已传真实尺寸。
- **修复:** 给两个 `SpotSelect/Spotselect` 加 `image_size` 参数并透传 `(height,width)`;
  `cuda` 改 `self._execution_device`。

### 4. `FLUX_kontext.generate(config=None)` 直接解引用 → `AttributeError`
- **位置:** `FLUX_kontext/flux_spotedit.py`(签名 `Optional[...] = None`,却直接读 `config.reset_steps` 等)
- **问题:** 不传 `config`(文档说合法)会在第 2 步崩;`flux_spotedit_tra.py` 同病。
- **修复:** `generate()` 开头加 `if config is None: config = SpotEditConfig()`。

### 5. `FLUX2` LPIPS 参考缓存从不失效 → 第 2 张及以后的图被拿去和第 1 张比
- **位置:** `FLUX2/FLUX2LPIPS.py:106`(`forward` 用 `_z2_feats_cache is not None` 判断)、
  `FLUX2/flux2_spot_ultis.py`(`SpotSelect` 仅在 `_z2_cached is None` 时 `set_z2_cache`)
- **问题:** `_check_z2_cache_valid` 是死代码(从未被调用)。metric 挂在 pipe 上、跨调用存活,
  同一个 `pipe` 处理第二张图时复用掩码用的是**第一张**图的特征。Qwen 路径用了
  `_check_z2_cache_valid` 会自愈,FLUX2 不会。
- **修复:** `forward()` 改用 `_check_z2_cache_valid(z2)` 比对并在不匹配时刷新缓存;或
  `generate()` 入口 `clear_cache()`。
- **状态:** 已确认 `_check_z2_cache_valid` 在 FLUX2 路径无调用点。

### 6. `qwen_plus` 静默忽略 CFG(评审追加,已验证)
- **位置:** `Qwen_image_edit_plus/qwen_plus_spotedit.py:134`(`do_true_cfg`)、`:144-153`
  (编码负向 prompt)、去噪循环 `:323-334`(只跑 `"cond"` 前向)
- **问题:** `do_true_cfg`、`negative_prompt`、`true_cfg_scale=4.0` 都接收并带 warning 校验,
  负向 prompt 甚至被编码,但去噪循环**只做单次 cond 前向**,负向/无条件分支**从未使用**。
  调用者若按原生 `QwenImageEditPlusPipeline`(默认 `true_cfg_scale=4.0` 且会做 CFG)的预期来用,
  会得到 prompt 跟随更弱的结果且**无报错**,还白白浪费一次负向编码。(这是"公平加速=不做 CFG"
  的有意设计,但 API 表面在误导。)
- **修复:** 要么补上无条件前向 + 合并(`pred = uncond + scale*(cond-uncond)`),要么删掉 CFG
  相关参数并在 docstring 注明只支持单前向。

---

## 🟡 中 — 真实,但属边角 / 非默认

### 7. `reset_steps` 写死 `[13,22,31]`,不随步数缩放
- **位置:** 4 份 `*_spot_ultis.py` 的 `SpotEditConfig`;`i in config.reset_steps` 各处
- **问题:** `FLUX_kontext` 默认 28 步 → 索引 31 永不触发;示例用 40 步;<14 步则一次 reset 都没有;
  >50 步则 reset 全挤在前 1/3。作者 notebook 自行覆盖为 `[11,19,27]`,印证需手调。
- **修复:** 由 `num_inference_steps` 推导(按比例或 every-K),或对传入列表做校验并对越界项告警。

### 7b. SDEdit 下 `reset_steps` 错位(评审追加,已验证)
- **位置:** `Qwen_image_edit_plus/qwen_plus_spotedit.py:247-254`
- **问题:** `source_init_strength<1.0` 时做了 `timesteps = timesteps[start_idx:]` 且 `i` 从 0
  重新计数,但 `reset_steps` 是相对**完整**调度的绝对索引 → reset 节奏错位。
- **修复:** reset 判定按 `start_idx` 偏移,或改用 by-fraction 推导。

### 8. `overwrite` 模式下 `boundary_aware_smoothing` 是空操作
- **位置:** `FLUX2/flux2_spotedit.py:291-296`、`Qwen_image_edit_plus/qwen_plus_spotedit.py:397-402`、
  `FLUX_kontext/flux_spotedit_tra.py`
- **问题:** 先把 ref 硬贴到**所有** `cache_final` 位置,再调平滑——此时边界处 `x_gen==ref`,
  混合 `λ·y+(1-λ)·x_gen` 塌缩回 `y`,本该消除的接缝原样保留。默认是 `velocity` 故影响有限,
  但 FLUX2-dev **唯一**合并路径就是这个。
- **修复:** 先保存硬贴前的生成 latent 作为 `x_gen`;函数内只对 interior 硬置 ref、对边界用硬贴前
  的 latent 做混合。

### 9. `FLUX_kontext.dilate_uncached_mask` 方向反了
- **位置:** `FLUX_kontext/flux_spot_ultis.py`(直接对 reuse 掩码 max-pool)
- **问题:** 其他三个是 反转→池化→反转(扩大*编辑*区,符合 docstring),FLUX_kontext 直接池化
  reuse 掩码 → 扩大*缓存*区、缩小编辑区,语义相反,边界会缓存到编辑区 → 接缝/欠编辑。
  目前因默认 `dilation_radius=0` 被掩盖;设 radius>0 就出问题(例如照搬 FLUX2 的 radius=1)。
- **修复:** 替换为标准 invert/dilate/invert 逻辑。

### 10. 默认 LPIPS 判据跑幻影 16 帧 VAE 解码(两个 Qwen backbone)
- **位置:** `Qwen_image_edit/QwenTokenLPIPS.py`、`Qwen_image_edit_plus/QwenTokenLPIPS.py`(两文件逐字节相同)
- **问题:** `_safe_unpack_tokens_2d` 返回 4D `(B,16,H,W)` 无时间轴,`_apply_qwen_mean_std`
  把 mean/std reshape 成 5D `(1,16,1,1,1)`;z_dim==C==16 时广播成 `(1,16,16,H,W)` 幻影 16 帧,
  被 3D 因果 VAE 解码(判据 ~16× 计算)再平均掉。结果正确,但默认判据白花 ~16×,且 batch>1 会崩。
  **`CHANGES.md` 描述的 `temporal_fix` 不在工作树里。**
- **修复:** `_safe_unpack_tokens_2d` 末尾 `z = z.unsqueeze(2)` 让 T=1,并设为默认;或默认改用
  免费的 cosine/L4。
- **状态:** 已独立确认。

### 11. 一个 `threshold` 被 3 种判据共用,量纲/方向不兼容
- **位置:** 4 份 `*_spot_ultis.py` 的 `SpotSelect/Spotselect`
- **问题:** `threshold`(FLUX/FLUX2=0.2,Qwen=0.15)原样传给所选判据,但 L4 用 `mean(|Δ|^4)<thr`
  (极小)、cosine 用 `sim>thr`(~0.9)、LPIPS 用 `score<thr`,量纲与方向都不同。`judge_method`
  作为开放旋钮,切换后会静默选错。`CHANGES.md` 自己也提到分数不可互换、需按目标复用比标定。
- **修复:** 每种方法各自 threshold 字段,或把分数归一到统一 [0,1] 复用概率再阈值化。

### 12. `FLUX_kontext.generate` 缺 `generator` 参数 + `prompt=None` 分支 `NameError`
- **位置:** `FLUX_kontext/flux_spotedit.py`
- **问题:** 不同于其他 backbone,无 `generator` 参数且 `prepare_latents(..., None, None)` 写死 →
  无法按调用确定种子(seam/seed 评测脚本失效)。另:`prompt=None` 回退分支引用了不存在的
  `prompt_embeds` 形参 → `NameError`。
- **修复:** 加 `generator` 参数并透传;删除死分支或补上 `prompt_embeds` 形参。

### 13. `requirements.txt` 完全不锁版本,而代码 import 了版本相关的 diffusers 内部模块
- **位置:** `requirements.txt`;`FLUX2/flux2_spotedit.py:9`、`FLUX2/flux2klein_spotedit.py:16`
- **问题:** dev 需 `diffusers>=0.36`(`pipeline_flux2`),klein 需 `>=0.37`(`pipeline_flux2_klein`);
  `FLUX2/__init__` 又急切 import,版本不符即 ImportError。直接用到的 `accelerate`/`Pillow` 也没列。
- **修复:** 锁定区间(如 `diffusers>=0.36,<0.40`、`torch>=2.9`、`transformers>=4.57`),
  补 `accelerate`/`Pillow`,并写明各 backbone 的版本矩阵。

### 14. 缺 `asset/` 目录 → 示例 notebook 崩、README 图裂
- **位置:** `README.md:3,26,54`;`examples/*.ipynb` 的 `load_image('./asset/dog.jpg')`
- **问题:** 仓库无 `asset/`,README 内嵌 `asset/*.jpg` 全裂,notebook happy path 第一步即 `FileNotFoundError`。
- **⚠️ 待确认:** 本地同步时**故意排除**了 `asset/`,请先在远端仓库确认它是否真不存在再处理。
- **修复:** 提交 `asset/`(dog.jpg + README 拼图),或改成可下载 URL / 明确占位并修 README 路径。

---

## ⚪ 低 / 提示(长尾)

| 项 | 位置 | 说明 |
|---|---|---|
| 死状态 `_cached_t` | 4 个 processor | 每步写、从不读(被移除的混合本来要用它) |
| `FLUX_kontext` LPIPS 缓存从不填充 | `FLUXLPIPS.py` | `set_z2_cache` 不设 `_z2_cached`,z2 每 reset 步解码 ~2 次 |
| cosine 判据返回 `[B*N]` 非 `[N]` | 3 处 `SpotSelect` | 边角;仅 B==1 路径,B>1 才触发 |
| Qwen `Spotselect` 无 `else` 兜底 | 两个 Qwen util | 未知 `judge_method` → `None` → `dilate` 处隐晦崩溃;FLUX2 会 raise |
| `x0_preds` 存全程 latent 仅读 `[-1]` | 5 个 generate | 每次调用内的瞬时内存浪费 |
| README 路径/引用拼写 | `README.md:43,44,62,65` | `example\flux.ipynb`(反斜杠+单数);`Ciatation`/`@artical` |
| `SpotSelect` vs `Spotselect` 大小写不一 | 跨 backbone | + `*_spot_ultis.py`("utils" 拼错)已固化进包名 |
| notebook 提交了报错输出 | `examples/qwen_plus_2511.ipynb` | 含 `ModuleNotFoundError` traceback |
| `debug_import.py` 写死个人路径 | `debug_import.py:2` | `/home/svu/e1352224/...`,他机静默 no-op |
| 死文件/死函数 | `flux_spotedit_tra.py`、`prune_isolated_caches` | 未被引用 |
| 可变默认参 `config=SpotEditConfig()` | 4 个 generate | 共享单例,改 `Optional=None` 内部构造更稳 |
| `select_every_step` 仅 qwen_plus 可配 | 跨 backbone | 复用掩码重算节奏不一致 |
| `boundary_aware_smoothing` 缺 `hw` 参数 | FLUX_kontext/Qwen | 仍假设方形 token 网格 |
| CPU 测试"8/8 逐位相同"措辞过强 | `test_spotedit_cpu.py` | 实为 `allclose(1e-6)`,仅覆盖 2/4 backbone 的 attention |
| `CHANGES.md` 引用了不在树里的脚本/文档 | `CHANGES.md` | `LPIPS_SPEED_STUDY.md` 等及"可由上述脚本复现"措辞 |

---

## 需要 GPU + 真权重才能确认

仅做静态分析:`velocity` 相对(惰性的)`overwrite` 是否真无缝;幻影帧的实际性能/质量损失量级;
FLUX2 跨图陈旧缓存是否真产生可见错误掩码;号称的 ~2× 加速;多参考图 Kontext 与非方图 LPIPS
崩溃(靠形状推演,未实跑);B>1 的 cosine 掩码 bug(当前 B==1 路径不可达)。

建议后续 GPU pass:每个 backbone 各跑一次 `generate()`(方图 + 非方图、默认与非默认
`judge_method`、同一 pipe 连续两张图),以确认崩溃/状态泄漏/缓存陈旧,并量化性能脚枪。

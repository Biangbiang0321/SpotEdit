# SpotEdit TODO(评审整改清单)

来源:[`CODE_REVIEW.md`](./CODE_REVIEW.md)(2026-06-29)。按优先级分组,勾选即完成。
图例:🔴 阻塞发布 · 🟠 常规路径出错 · 🟡 边角/非默认 · ⚪ 长尾。

---

## P0 — 发布闸门(必须先修)

- [ ] **🔴 FLUX_kontext 改相对 import**
  `flux_spotedit.py:15-16`、`flux_spot_ultis.py:15`、`flux_spotedit_tra.py:15-16`
  → `from .flux_spot_ultis import …` / `from .fluxSpotAttn import …` / `from .FLUXLPIPS import …`

- [ ] **🟠 processor 安装包 try/finally 还原**(4 个 backbone)
  给 `Qwen_image_edit`、`FLUX_kontext` 补 `_orig_procs` 快照 + 还原;
  给 `FLUX2`、`qwen_plus` 的还原套上 `try/…finally:`,确保异常/OOM 也还原。

- [ ] **🟠 LPIPS 判据透传真实 image_size + 设备**
  `FLUX_kontext.SpotSelect`、`Qwen_image_edit.Spotselect` 加 `image_size` 参数并传 `(height,width)`;
  `self.vae.to('cuda')` → `self.vae.to(self._execution_device)`。

- [ ] **🟠 FLUX_kontext `config=None` 归一化**
  `flux_spotedit.py`(及 `flux_spotedit_tra.py`)`generate()` 开头加
  `if config is None: config = SpotEditConfig()`。

- [ ] **🟠 FLUX2 LPIPS 缓存按 z2 失效**
  `FLUX2/FLUX2LPIPS.py:forward` 改用 `_check_z2_cache_valid(z2)` 判断,不匹配则刷新;
  或 `generate()` 入口 `metric.clear_cache()`。

- [ ] **🟠 qwen_plus CFG:实现或删除**
  `qwen_plus_spotedit.py`:要么补无条件前向 + `uncond+scale*(cond-uncond)` 合并,
  要么删除 `true_cfg_scale`/`negative_prompt*` 参数与 `do_true_cfg` 分支并在 docstring 注明单前向。

---

## P1 — 应修(真实缺陷,边角/非默认)

- [ ] **🟡 reset_steps 随步数推导**(4 份 config)
  由 `num_inference_steps` 按比例/every-K 推导,或对传入列表校验越界并告警。

- [ ] **🟡 SDEdit 下 reset_steps 偏移**
  `qwen_plus_spotedit.py:247-254`:reset 判定按 `start_idx` 偏移(或改 by-fraction)。

- [ ] **🟡 修 overwrite 模式 boundary_aware_smoothing 空操作**
  `flux2_spotedit.py:291-296`、`qwen_plus_spotedit.py:397-402`:先存硬贴前 latent 作 `x_gen`,
  函数内只对 interior 硬置 ref、边界用硬贴前 latent 混合。

- [ ] **🟡 FLUX_kontext dilate 方向修正**
  `flux_spot_ultis.py`:换成 invert→max_pool→invert(与其他三个一致)。

- [ ] **🟡 Qwen LPIPS 去幻影 16 帧**
  两个 `QwenTokenLPIPS.py` 的 `_safe_unpack_tokens_2d` 末尾 `z = z.unsqueeze(2)`(T=1);
  设为默认或默认改 cosine/L4。

- [ ] **🟡 threshold 按判据分离**
  每种 `judge_method` 各自 threshold,或分数归一到 [0,1] 复用概率再阈值化。

- [ ] **🟡 FLUX_kontext 加 generator 参数 + 修 prompt=None 分支**
  加 `generator` 并透传 `prepare_latents`;删除/修正引用未定义 `prompt_embeds` 的死分支。

- [ ] **🟡 requirements.txt 锁版本**
  `diffusers>=0.36,<0.40`、`torch>=2.9`、`transformers>=4.57`,补 `accelerate`/`Pillow`,
  写明各 backbone 版本矩阵。

- [ ] **🟡 确认并补齐 asset/**(⚠️ 先确认远端是否真缺)
  补 `asset/`(dog.jpg + README 拼图),或 notebook 改可下载 URL 并修 README 图路径。

---

## P2 — 清理与一致性(长尾)

- [ ] ⚪ 移除 4 个 processor 的死状态 `_cached_t`
- [ ] ⚪ FLUX_kontext `FLUXLPIPS.set_z2_cache` 设 `_z2_cached`(让缓存生效,省 ~2× 解码)
- [ ] ⚪ cosine 判据按 batch 归约,返回 `[N]`(3 处 `SpotSelect`)
- [ ] ⚪ Qwen `Spotselect` 加 `else: raise ValueError(...)` 兜底未知 `judge_method`
- [ ] ⚪ `x0_preds` 列表改成只存 `last_x0` 单变量(5 个 generate)
- [ ] ⚪ 统一判据命名 `SpotSelect`;模块重命名 `*_spot_utils.py`(改 import/__init__)
- [ ] ⚪ `config=SpotEditConfig()` 默认参 → `Optional=None` 内部构造(4 个 generate)
- [ ] ⚪ `select_every_step`、`boundary_aware_smoothing(hw=...)` 回填到 FLUX_kontext/Qwen
- [ ] ⚪ 删死文件 `flux_spotedit_tra.py`、死函数 `prune_isolated_caches`、未用 import/多余 clone

## P3 — 文档与仓库卫生

- [ ] ⚪ README 修路径 `examples/flux.ipynb`(去反斜杠/单数)+ `Citation`/`@article`
- [ ] ⚪ 清空 notebook 输出(尤其 `qwen_plus_2511.ipynb` 的报错 traceback);加 nbstripout/CI
- [ ] ⚪ `debug_import.py` 改用相对自身路径,去掉个人集群绝对路径
- [ ] ⚪ CPU 测试措辞改"~1e-6 近似(仅 FLUX2/qwen_plus attention)",注明未覆盖项
- [ ] ⚪ `CHANGES.md` 去掉"可由上述脚本复现"或标注脚本为内部/已移除

---

## 仅静态分析、需 GPU + 真权重确认

- [ ] velocity vs overwrite 实际无缝性、~2× 加速、幻影帧性能/质量损失量级
- [ ] FLUX2 跨图缓存陈旧是否产生可见错误掩码
- [ ] 多参考图 Kontext、非方图 LPIPS、B>1 cosine 掩码 的实跑验证
- [ ] 回归基线:整改后跑 `python test_spotedit_cpu.py` 应保持 8/8

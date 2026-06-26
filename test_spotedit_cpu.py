"""
CPU correctness tests for the SpotEdit FLUX2 / Qwen_image_edit_plus modules.
No GPU or model weights needed -- uses tiny randomly-initialised real diffusers
modules. Run from the `spotedit/` dir:  python test_spotedit_cpu.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

torch.manual_seed(0)
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, extra=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}{('  | ' + extra) if extra else ''}")


# ----------------------------------------------------------------------------
# 1. no-cache step == stock diffusers processor (forward path unchanged)
# ----------------------------------------------------------------------------
def test_equivalence():
    print("1. SpotEdit processor (no-cache step) == stock diffusers processor")
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Attention, Flux2ParallelSelfAttention, Flux2AttnProcessor,
        Flux2ParallelSelfAttnProcessor, Flux2PosEmbed)
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
    from FLUX2.flux2SpotAttn import Flux2SpotAttnProcessor, Flux2ParallelSpotAttnProcessor
    from Qwen_image_edit_plus.qwenSpotAttn import QwenSpotEditAttnProcessor

    dim, heads, hd = 64, 2, 32
    text_n, latent_n2, image_n = 3, 4, 4
    full = lambda: [torch.zeros(text_n, dtype=torch.bool), torch.zeros(latent_n2, dtype=torch.bool),
                    torch.zeros(image_n, dtype=torch.bool), 1.0]
    pos = Flux2PosEmbed(theta=2000, axes_dim=(8, 8, 8, 8))
    ids = lambda n, t0: torch.tensor([[t0, k // max(int(n**0.5),1), k % max(int(n**0.5),1), 0] for k in range(n)], dtype=torch.float32)
    tc, ts_ = pos(ids(text_n, 0)); ic, isn = pos(torch.cat([ids(latent_n2, 1), ids(image_n, 2)], 0))
    rope = (torch.cat([tc, ic], 0), torch.cat([ts_, isn], 0))
    eq = lambda a, b: bool(torch.allclose(a, b, atol=1e-6, rtol=0))

    torch.manual_seed(1)
    d = Flux2Attention(query_dim=dim, added_kv_proj_dim=dim, dim_head=hd, heads=heads, out_dim=dim,
                       bias=False, added_proj_bias=False, out_bias=False, eps=1e-5).eval()
    img = torch.randn(1, latent_n2 + image_n, dim); txt = torch.randn(1, text_n, dim)
    with torch.no_grad():
        d.set_processor(Flux2AttnProcessor()); sh, se = d(img, txt, image_rotary_emb=rope)
        d.set_processor(Flux2SpotAttnProcessor(full())); ph, pe = d(img, txt, image_rotary_emb=rope)
    check("FLUX.2 double  img", eq(sh, ph)); check("FLUX.2 double  txt", eq(se, pe))

    torch.manual_seed(2)
    s = Flux2ParallelSelfAttention(query_dim=dim, dim_head=hd, heads=heads, out_dim=dim,
                                   bias=False, out_bias=False, eps=1e-5, mlp_ratio=4.0, mlp_mult_factor=2).eval()
    hs = torch.randn(1, text_n + latent_n2 + image_n, dim)
    with torch.no_grad():
        s.set_processor(Flux2ParallelSelfAttnProcessor()); so = s(hs, image_rotary_emb=rope)
        s.set_processor(Flux2ParallelSpotAttnProcessor(full())); po = s(hs, image_rotary_emb=rope)
    check("FLUX.2 single  parallel", eq(so, po))

    torch.manual_seed(3)
    q = Attention(query_dim=dim, cross_attention_dim=None, added_kv_proj_dim=dim, dim_head=hd, heads=heads,
                  out_dim=dim, context_pre_only=False, bias=True, qk_norm="rms_norm", eps=1e-6,
                  processor=QwenDoubleStreamAttnProcessor2_0()).eval()
    qhs = torch.randn(1, latent_n2 + image_n, dim); qenc = torch.randn(1, text_n, dim)
    qmask = torch.ones(1, text_n, dtype=torch.long)
    with torch.no_grad():
        qi_s, qt_s = q(qhs, encoder_hidden_states=qenc, encoder_hidden_states_mask=qmask, image_rotary_emb=None)
        q.set_processor(QwenSpotEditAttnProcessor(full()))
        qi_p, qt_p = q(qhs, encoder_hidden_states=qenc, encoder_hidden_states_mask=qmask, image_rotary_emb=None)
    check("Qwen   double  img", eq(qi_s, qi_p)); check("Qwen   double  txt", eq(qt_s, qt_p))


# ----------------------------------------------------------------------------
# 2. cached step reusing UNCHANGED tokens == full recompute (core claim)
# ----------------------------------------------------------------------------
def test_cached_exactness():
    print("2. FLUX.2 cached step (reusing unchanged tokens) == full recompute, all layers")
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Transformer2DModel, Flux2Attention, Flux2ParallelSelfAttention)
    from FLUX2.flux2SpotAttn import Flux2SpotAttnProcessor, Flux2ParallelSpotAttnProcessor

    inC, jdim, hd, nh = 16, 24, 32, 2
    m = Flux2Transformer2DModel(patch_size=1, in_channels=inC, num_layers=2, num_single_layers=2,
                                attention_head_dim=hd, num_attention_heads=nh, joint_attention_dim=jdim,
                                timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(8, 8, 8, 8),
                                rope_theta=2000, eps=1e-6).eval()
    text_n, latent_n2, image_n = 3, 4, 4
    ids = lambda n, t0: torch.tensor([[t0, k // max(int(n**0.5),1), k % max(int(n**0.5),1), 0] for k in range(n)], dtype=torch.float32)
    txt_ids = ids(text_n, 0); img_ids = torch.cat([ids(latent_n2, 1), ids(image_n, 2)], 0)
    prompt = torch.randn(1, text_n, jdim); latents = torch.randn(1, latent_n2, inC); image_latents = torch.randn(1, image_n, inC)
    ts = torch.tensor([0.5]); guid = torch.tensor([4.0])

    def install(cf):
        for _, mod in m.named_modules():
            if isinstance(mod, Flux2Attention): mod.set_processor(Flux2SpotAttnProcessor(cf))
            elif isinstance(mod, Flux2ParallelSelfAttention): mod.set_processor(Flux2ParallelSpotAttnProcessor(cf))

    reuse = torch.tensor([True, True, False, False])      # reuse tokens 0,1 ; recompute 2,3
    cf = [torch.zeros(text_n, dtype=torch.bool), torch.zeros(latent_n2, dtype=torch.bool),
          torch.zeros(image_n, dtype=torch.bool), 1.0]
    install(cf)
    with torch.no_grad():
        out_full = m(hidden_states=torch.cat([latents, image_latents], 1), timestep=ts, guidance=guid,
                     encoder_hidden_states=prompt, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
        # cached step: image fully cached, feed only the unchanged uncached latents
        cf[1] = reuse; cf[2] = torch.ones(image_n, dtype=torch.bool); cf[3] = 1.0
        out_cached = m(hidden_states=latents[:, ~reuse], timestep=ts, guidance=guid,
                       encoder_hidden_states=prompt, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    ref = out_full[:, :latent_n2][:, ~reuse]              # full-step output at the recomputed latent slots
    d = (out_cached - ref).abs().max().item()
    check("cached(unchanged reuse) == full recompute", torch.allclose(out_cached, ref, atol=1e-5, rtol=0), f"max|Δ|={d:.2e}")


# ----------------------------------------------------------------------------
# 3. FLUX.2 LPIPS unpack/denorm/unpatchify is the exact inverse of the pack path
# ----------------------------------------------------------------------------
def test_lpips_roundtrip():
    print("3. FLUX.2 token-LPIPS unpack == exact inverse of pipeline pack/patchify/bn")
    import torch.nn as nn
    from types import SimpleNamespace
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline as P
    from FLUX2.FLUX2LPIPS import FLUX2VAETokenLPIPS
    C, Hl, Wl = 4, 8, 8
    z = torch.randn(1, C, Hl, Wl)
    z_p = P._patchify_latents(z); Ctok = z_p.shape[1]
    mean = torch.randn(Ctok); var = torch.rand(Ctok) + 0.5; eps = 1e-4
    z_pn = (z_p - mean.view(1, -1, 1, 1)) / torch.sqrt(var + eps).view(1, -1, 1, 1)
    tokens = P._pack_latents(z_pn)
    bn = nn.BatchNorm2d(Ctok); bn.running_mean.copy_(mean); bn.running_var.copy_(var)
    dec = nn.Module(); dec.register_parameter("p", nn.Parameter(torch.zeros(1)))
    vae = SimpleNamespace(bn=bn, decoder=dec, config=SimpleNamespace(batch_norm_eps=eps))
    m = FLUX2VAETokenLPIPS(vae)
    z2 = m._unpatchify(m._apply_bn_denorm(m._safe_unpack_tokens(tokens, (64, 64), 8)))
    check("LPIPS unpack roundtrip", torch.allclose(z2, z, atol=1e-5), f"max|Δ|={(z2 - z).abs().max().item():.2e}")


# ----------------------------------------------------------------------------
# 4. KV reassembly handles multiple reference images (image_n != latent_n2)
# ----------------------------------------------------------------------------
def test_multi_image_reassembly():
    print("4. KV reassembly with multiple reference images (image_n != latent_n2)")
    from types import SimpleNamespace
    from FLUX2.flux2SpotAttn import _reassemble_kv
    text_n, latent_n2, image_n, Hh, Dd = 2, 4, 6, 3, 5
    reuse = torch.tensor([True, False, True, False])
    cf = [torch.zeros(text_n, dtype=torch.bool), reuse, torch.ones(image_n, dtype=torch.bool), 0.5]
    full = torch.arange(text_n + latent_n2 + image_n).float().view(1, -1, 1, 1).expand(1, -1, Hh, Dd).clone()
    proc = SimpleNamespace(cache_flags=cf, _cached_keys=full.clone(), _cached_values=full.clone() + 100,
                           _cached_t=torch.ones(1, text_n + latent_n2 + image_n))
    un = int((~reuse).sum())
    key_in = torch.cat([full[:, :text_n], torch.full((1, un, Hh, Dd), 99.0)], dim=1)
    k, _ = _reassemble_kv(proc, key_in, key_in + 100, text_n, un, latent_n2)
    ok = (k.shape[1] == text_n + latent_n2 + image_n
          and torch.equal(k[:, :text_n], full[:, :text_n])
          and torch.allclose(k[:, text_n:text_n + latent_n2][:, ~reuse], torch.full((1, un, Hh, Dd), 99.0))
          and torch.allclose(k[:, text_n:text_n + latent_n2][:, reuse], full[:, text_n:text_n + latent_n2][:, reuse])
          and torch.equal(k[:, text_n + latent_n2:], full[:, text_n + latent_n2:]))
    check("multi-image reassembly (len/text/uncached/reused/image)", ok, f"out len={k.shape[1]}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    test_equivalence()
    test_cached_exactness()
    test_lpips_roundtrip()
    test_multi_image_reassembly()
    print(f"\n{'='*60}\n{'ALL PASSED' if all(results) else 'SOME FAILED'}: {sum(results)}/{len(results)} checks\n{'='*60}")
    sys.exit(0 if all(results) else 1)

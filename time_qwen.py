import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from Qwen_image_edit_plus import generate, SpotEditConfig

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PROMPT = "add a red knitted scarf around the dog's neck"

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16).to("cuda")
img = load_image("asset/dog.jpg").resize((1024, 1024))

# ---- instrumentation: time text-encoder, transformer (per call + tokens), vae.decode ----
S = {"enc": [], "tf_t": [], "tf_tok": [], "dec": []}


def sync():
    torch.cuda.synchronize()


def tf_pre(m, args, kwargs):
    sync(); m.__t0 = time.perf_counter()
    hs = kwargs.get("hidden_states", args[0] if args else None)
    S["tf_tok"].append(int(hs.shape[1]) if hs is not None else -1)


def tf_post(m, args, kwargs, out):
    sync(); S["tf_t"].append(time.perf_counter() - m.__t0)


def enc_pre(m, args, kwargs):
    sync(); m.__t0 = time.perf_counter()


def enc_post(m, args, kwargs, out):
    sync(); S["enc"].append(time.perf_counter() - m.__t0)


pipe.transformer.register_forward_pre_hook(tf_pre, with_kwargs=True)
pipe.transformer.register_forward_hook(tf_post, with_kwargs=True)
pipe.text_encoder.register_forward_pre_hook(enc_pre, with_kwargs=True)
pipe.text_encoder.register_forward_hook(enc_post, with_kwargs=True)

_orig_decode = pipe.vae.decode
def timed_decode(*a, **k):
    sync(); t0 = time.perf_counter()
    r = _orig_decode(*a, **k)
    sync(); S["dec"].append(time.perf_counter() - t0)
    return r
pipe.vae.decode = timed_decode


def reset():
    for v in S.values():
        v.clear()


def report(name, total):
    enc = sum(S["enc"]); tf = sum(S["tf_t"]); dec = sum(S["dec"])
    rest = total - enc - tf - dec
    tok = S["tf_tok"]
    print(f"\n=== {name} ===", flush=True)
    print(f"  TOTAL            {total:6.2f}s", flush=True)
    print(f"  text_encode      {enc:6.2f}s  ({len(S['enc'])} calls)", flush=True)
    print(f"  transformer      {tf:6.2f}s  ({len(S['tf_t'])} calls)  tokens: min={min(tok)} mean={sum(tok)//len(tok)} max={max(tok)}", flush=True)
    print(f"  vae.decode       {dec:6.2f}s  ({len(S['dec'])} calls)", flush=True)
    print(f"  rest (sel/compose/sched/overhead) {rest:6.2f}s", flush=True)
    print(f"  per-call transformer tokens: {tok}", flush=True)
    print(f"  per-call transformer secs:   {[round(x,3) for x in S['tf_t']]}", flush=True)
    return {"total": total, "enc": enc, "tf": tf, "dec": dec, "rest": rest}


# ---- BASE (no SpotEdit, single forward to match SpotEdit compute) ----
reset()
torch.manual_seed(42); sync(); t0 = time.perf_counter()
pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0)
sync(); base = report("BASE (no SpotEdit, true_cfg=1.0)", time.perf_counter() - t0)

# ---- CONTROL: my generate() with caching OFF (initial_steps>=STEPS -> every step full) ----
# If this ~matches the stock base, the speedup is purely from caching, not code differences.
reset()
cfg_off = SpotEditConfig(initial_steps=STEPS + 1, reset_steps=[])
torch.manual_seed(42); sync(); t0 = time.perf_counter()
generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg_off)
sync(); ctrl = report("CONTROL: generate() caching OFF", time.perf_counter() - t0)

# ---- SpotEdit (velocity, caching ON) ----
reset()
cfg = SpotEditConfig(threshold=0.15, initial_steps=4, reset_steps=list(range(8, STEPS, RES)), dilation_radius=2)
torch.manual_seed(42); sync(); t0 = time.perf_counter()
generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg)
sync(); spot = report(f"SPOTEDIT velocity (reset every {RES})", time.perf_counter() - t0)

print("\n=== SPEEDUP ===", flush=True)
print(f"  stock base total            {base['total']:6.2f}s", flush=True)
print(f"  generate() caching-OFF total {ctrl['total']:6.2f}s   (== base? confirms no code-diff confound)", flush=True)
print(f"  SpotEdit caching-ON total   {spot['total']:6.2f}s", flush=True)
print(f"  speedup vs stock base:        {base['total']/spot['total']:.2f}x", flush=True)
print(f"  speedup vs same-code no-cache:{ctrl['total']/spot['total']:.2f}x   <-- the true caching speedup", flush=True)
print(f"  transformer-only (no-cache -> cache): {ctrl['tf']:.1f}s -> {spot['tf']:.1f}s = {ctrl['tf']/spot['tf']:.2f}x", flush=True)
print("[done]", flush=True)

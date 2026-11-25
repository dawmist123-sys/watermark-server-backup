"""50-step adapters for archived author watermark implementations."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline

from attacks import MAIN, EXTRA, attack
from benchmark import roc, test_rows


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf8")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", required=True, choices=("treering","grow","gaussmarker","gaussianshading","prc"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--author-source", type=Path, help="Override the archived author checkout")
    ap.add_argument("--gnr-checkpoint", type=Path)
    ap.add_argument("--gnr-key", type=Path)
    ap.add_argument("--gnr-classifier", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--attacks", nargs="+", choices=MAIN+EXTRA, default=MAIN)
    a = ap.parse_args()
    root = a.project_root.resolve()
    base = root / "revision_20261111"
    rows = test_rows(a.manifest, a.limit)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(base/"isolated_common"))
    pipe = StableDiffusionPipeline.from_pretrained(
        a.model, torch_dtype=torch.float16, local_files_only=True,
        safety_checker=None, requires_safety_checker=False,
        **({"variant": a.variant} if a.variant else {})).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    inversion = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    method = a.algorithm
    default_sources = {
        'treering': root/'tree-ring-watermark-main',
        'prc': base/'baselines/PRC-Watermark',
        'grow': base/'baselines/GROW/GROW',
        'gaussmarker': base/'baselines/GaussMarker',
        'gaussianshading': base/'baselines/GaussMarker',
    }
    author = (a.author_source or default_sources[method]).resolve()
    if not author.is_dir():
        ap.error(f"Author source directory not found: {author}")
    attacks = a.attacks
    notes=[]; wm=None; grow=None; restorer=None; classifier=None; tr=None; trmask=None; trargs=None
    torch.manual_seed(999999); np.random.seed(999999)
    if method=='grow':
        sys.path.insert(0,str(author)); from grow import GrowConfig,GrowWatermarker
        if pipe.scheduler.config.prediction_type=='v_prediction':
            alphas=pipe.scheduler.alphas_cumprod.to('cuda')
            def convert(module,args,output):
                v=alphas[args[1]]; raw=output[0] if isinstance(output,tuple) else output.sample
                converted=v.sqrt()*raw+(1-v).sqrt()*args[0]
                return (converted,)+output[1:] if isinstance(output,tuple) else type(output)(sample=converted)
            pipe.unet.register_forward_hook(convert); pipe.scheduler.register_to_config(prediction_type='epsilon'); inversion=DDIMInverseScheduler.from_config(pipe.scheduler.config)
            notes.append('Model v output converted algebraically to epsilon before author equations; scheduler correspondingly set to epsilon. Author source unchanged.')
        grow=GrowWatermarker(GrowConfig(model_id=a.model,dtype='float16',num_inference_steps=50,message='OK',secret_key='revision-20261111'),pipe=pipe)
        notes.append('Author utility implements real FFT as DCT proxy; kept unchanged and disclosed. Detection score is target payload bit agreement, not raw message accuracy reported as AUC.')
    elif method=='prc':
        sys.path.insert(0,str(author))
        from src.prc import KeyGen,Encode
        from src import pseudogaussians
        encoding_key,decoding_key=KeyGen(4*64*64,false_positive_rate=.01,t=3)
        import pickle
        with (out/'prc_keys.pkl').open('wb') as f:pickle.dump((encoding_key,decoding_key),f)
        notes.append('Author PRC keys, encoding and pseudogaussian sampling. Continuous score is the standardized margin in author Detect, rather than a fixed sign surrogate.')
    else:
        path=author/('optim_utils.py' if method=='treering' else 'tr_utils.py')
        spec=importlib.util.spec_from_file_location('author_tr_'+method,path); trmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(trmod)
        trargs=SimpleNamespace(w_seed=999999,w_channel=0 if method=='treering' else 3,w_pattern='ring',w_radius=10 if method=='treering' else 4,w_mask_shape='circle',w_measurement='l1_complex',w_injection='complex',w_pattern_const=0)
        tr=trmod.get_watermarking_pattern(pipe,trargs,'cuda',shape=(1,4,64,64)); trmask=trmod.get_watermarking_mask(tr.real,trargs,'cuda')
        torch.save({'template':tr.cpu(),'mask':trmask.cpu(),'args':vars(trargs)},out/'ring_key.pt')
        if method in ['gaussmarker','gaussianshading']:
            sys.path.insert(0,str(author)); from watermark import Gaussian_Shading_chacha
            if method=='gaussmarker':
                from unet.unet_model import UNet
                checkpoint = a.gnr_checkpoint or base/'GNR_256bits/model_final.pth'
                keyfile = a.gnr_key or base/'GNR_watermark_256.pth'
                assert checkpoint.exists() and keyfile.exists(),'GNR checkpoint and matching key required'
                info=torch.load(keyfile,map_location='cpu',weights_only=False)
                wm=Gaussian_Shading_chacha(1,8,8,.01,1,watermark=info['w'],m=info['m'],key=info['key'],nonce=info['nonce'])
                restorer=UNet(4,4,nf=128).cuda().eval(); restorer.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                import joblib
                classifier=joblib.load(a.gnr_classifier or author/'sd21_cls2.pkl')
                notes.append('Author GNR trained for 50000 steps on synthetic latent augmentations; author SD2.1 ensemble classifier reused; model mismatch, if any, must be disclosed.')
            else:wm=Gaussian_Shading_chacha(1,8,8,.01,1); wm.create_watermark_and_return_w_m()
            torch.save({'w':wm.watermark,'m':wm.m,'key':wm.key,'nonce':wm.nonce},out/'watermark_key.pth')
    manifest = {"args":vars(a), "steps":50, "generation_cfg":7.5, "inversion_cfg":1,
                "test_ids":[int(row["image_id"]) for row in rows], "notes":notes,
                "algorithm":method, "author_source":str(author), "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out/"manifest.json",manifest)
    def text(prompt):
        t=pipe.tokenizer(prompt,padding='max_length',truncation=True,max_length=pipe.tokenizer.model_max_length,return_tensors='pt'); return pipe.text_encoder(t.input_ids.cuda())[0]
    @torch.inference_mode()
    def invert(image):
        x=torch.from_numpy(np.array(image).copy()).permute(2,0,1).unsqueeze(0).to('cuda',torch.float16)/127.5-1; z=pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor
        emb=text(''); inversion.set_timesteps(50,device='cuda')
        for t in inversion.timesteps:
            inputs = inversion.scale_model_input(z,t)
            z = inversion.step(pipe.unet(inputs,t,encoder_hidden_states=emb).sample,t,z).prev_sample
        assert torch.isfinite(z).all(); return z
    @torch.inference_mode()
    def score(image):
        if method=='grow':
            from grow.codec import message_to_bits
            bits=grow._read_bits(image); target=message_to_bits(grow.cfg.message); return float(np.mean(np.array(bits)==np.array(target)))
        z=invert(image)
        if method=='treering':return -float((torch.fft.fftshift(torch.fft.fft2(z.float()),dim=(-2,-1))-tr)[trmask].abs().mean())
        if method=='prc':
            posterior=torch.as_tensor(pseudogaussians.recover_posteriors(z.flatten().double().cpu()))
            _,checks,pad,_,noise,_,_,_,degree=decoding_key
            q=(1-2*noise)*(1-2*np.array(pad,dtype=float))*posterior.numpy(force=True)
            product=np.clip(np.prod(q[checks.indices.reshape(checks.shape[0],degree)],axis=1),-1+1e-12,1-1e-12)
            plus=np.log((1+product)/2); minus=np.log((1-product)/2); both=plus+minus
            constant=.5*np.sum(plus**2+minus**2-.5*both**2)
            return float((plus.sum()-.5*both.sum())/np.sqrt(max(2*constant,1e-16)))
        if method=='gaussianshading':return float((wm.pred_w_from_latent(z).cpu()==wm.watermark.cpu()).float().mean())
        restored=(restorer((z>0).float()).sigmoid()>.5).int().cpu(); decoded=wm.pred_w_from_m(restored); agreement=float((decoded.cpu()==wm.watermark.cpu()).float().mean())
        ring=-float((torch.fft.fftshift(torch.fft.fft2(z.float()),dim=(-2,-1))-tr)[trmask].abs().mean())
        return float(classifier.predict_proba([[agreement,ring]])[0,1])
    started = time.time()
    scores = {name: ([], []) for name in attacks}
    generation_times, detection_times = [], []
    for j, row in enumerate(rows):
        image_id, seed, prompt = int(row["image_id"]), int(row["seed"]), row["caption"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn((1, 4, 64, 64), generator=generator).to("cuda", torch.float16)
        with torch.inference_mode():
            clean = pipe(prompt, latents=z.clone(), num_inference_steps=50,
                         guidance_scale=7.5, height=512, width=512, eta=0).images[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if method == 'grow':
            marked = grow.generate_with_watermark(prompt, seed=seed)
        else:
            if method == 'prc':
                latent = pseudogaussians.sample(Encode(encoding_key)).reshape(1, 4, 64, 64).to('cuda', torch.float16)
            elif method == 'treering':
                latent = trmod.inject_watermark(z.float(), trmask, tr, trargs).half()
            else:
                latent = wm.create_watermark_and_return_w_m()[0].cuda().half()
                if method == 'gaussmarker':
                    latent = trmod.inject_watermark(latent.float(), trmask, tr, trargs).half()
            with torch.inference_mode():
                marked = pipe(prompt, latents=latent, num_inference_steps=50,
                              guidance_scale=7.5, height=512, width=512, eta=0).images[0]
        torch.cuda.synchronize()
        generation_times.append(time.perf_counter()-t0)
        folder = out/'images'/str(image_id)
        folder.mkdir(parents=True)
        clean.save(folder/'clean.png')
        marked.save(folder/'marked.png')
        with (out/'scores.jsonl').open('a', encoding='utf8') as stream:
            for name, pair in scores.items():
                for label, source in enumerate((clean, marked)):
                    image = attack(source, name, seed)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    value = score(image)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-t0
                    if not np.isfinite(value):
                        raise RuntimeError('Non-finite score')
                    pair[label].append(value)
                    if name == 'clean' and label == 1:
                        detection_times.append(elapsed)
                    stream.write(json.dumps(dict(image_id=image_id, condition=name, label=label,
                                                 score=value, detection_seconds=elapsed))+'\n')
        print(j+1, '/', len(rows), flush=True)
    summary = dict(method=method, count=len(rows), conditions={k: roc(*v) for k, v in scores.items()},
                   elapsed_seconds=time.time()-started)
    for name, values in (('generation', generation_times), ('detection', detection_times)):
        if values:
            summary[name+'_seconds'] = dict(mean=float(np.mean(values)),
                                             sd=float(np.std(values, ddof=1)) if len(values)>1 else None)
    write_json(out/'summary.json', summary)


if __name__ == "__main__":
    main()

"""Trains the small swap model to copy inswapper_128 (knowledge distillation).

Each step: take a batch of aligned face crops (the "camera" faces) and a batch of other people's
identities, run the big model (teacher) to get its swapped faces, and train the small model (student)
to produce the same pictures, while also matching the requested identity.

  python distill.py --data /data/prepared --models /models --out /runs/fast1 --steps 80000 --hours 10

Writes in --out: train_log.csv, samples/step_XXXXXX.jpg, ckpt_last.pt, ckpt_best.pt, val.jsonl.
Resumes from ckpt_last.pt if it exists.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx2torch import convert

sys.path.insert(0, os.path.dirname(__file__))
from faces import emap_from_swapper  # noqa: E402
from student import Student, gflops  # noqa: E402


def load_shards(folder, max_faces=0):
    crops, embs = [], []
    for f in sorted(glob.glob(os.path.join(folder, 'shard_*.npz'))):
        d = np.load(f)
        crops.append(d['crops'])
        embs.append(d['emb'].astype(np.float32))
        if max_faces and sum(len(c) for c in crops) >= max_faces:
            break
    crops = np.concatenate(crops)[: max_faces or None]
    embs = np.concatenate(embs)[: max_faces or None]
    return crops, embs


def jitter(x, strength):
    """Small random shifts, turns and zooms of the aligned crops, like the app's tracker-based alignment."""
    if strength <= 0:
        return x
    n = x.shape[0]
    ang = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(4) * strength
    scale = 1 + (torch.rand(n, device=x.device) * 2 - 1) * 0.05 * strength
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.03 * strength
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def arc_input(x128):
    """Swap-layout 128 crop (0..1) -> ArcFace 112 crop (-1..1): the 128 layout is the 112 one shifted 8px right."""
    return x128[:, :, 0:112, 8:120] * 2 - 1


def psnr(a, b):
    mse = F.mse_loss(a, b, reduction='none').mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='output folder of prepare_data.py')
    ap.add_argument('--models', required=True, help='folder with inswapper_128.onnx and w600k_r50.onnx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=256, help='student channels at 32x32 (256 ~21 GFLOPs, 384 ~40)')
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--steps', type=int, default=80000)
    ap.add_argument('--hours', type=float, default=0, help='stop (and save) after this long; 0 = no limit')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--w-l1', type=float, default=1.0)
    ap.add_argument('--w-lpips', type=float, default=0.5)
    ap.add_argument('--w-id', type=float, default=0.15)
    ap.add_argument('--jitter', type=float, default=1.0)
    ap.add_argument('--val-every', type=int, default=2000)
    ap.add_argument('--ckpt-every', type=int, default=5000)
    ap.add_argument('--max-faces', type=int, default=0)
    ap.add_argument('--val-faces', type=int, default=1000, help='faces kept out of training for validation')
    ap.add_argument('--val-pairs', type=int, default=256)
    ap.add_argument('--workers-seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.workers_seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dict(device_type='cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else dict(device_type='cpu', enabled=False)
    os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    crops, embs = load_shards(args.data, args.max_faces)
    n_val = min(args.val_faces, len(crops) // 10)
    print(f'{len(crops)} faces ({len(crops) - n_val} training, {n_val} validation) on {dev}', flush=True)
    # Faces live on the GPU as uint8 (120k faces ~ 5.9 GB); fine on a 24 GB card, else keep them on the CPU.
    keep_on_gpu = dev.type == 'cuda' and crops.nbytes < 0.4 * torch.cuda.get_device_properties(0).total_memory
    store = torch.from_numpy(crops).to(dev if keep_on_gpu else 'cpu')
    emb_all = torch.from_numpy(embs).to(dev)
    train_idx = torch.arange(n_val, len(crops))
    val_idx = torch.arange(0, n_val)

    emap = torch.from_numpy(emap_from_swapper(os.path.join(args.models, 'inswapper_128.onnx'))).to(dev)
    teacher = convert(onnx.load(os.path.join(args.models, 'inswapper_128.onnx'))).to(dev).eval()
    arcface = convert(onnx.load(os.path.join(args.models, 'w600k_r50.onnx'))).to(dev).eval()
    import gc

    gc.collect()
    for p in list(teacher.parameters()) + list(arcface.parameters()):
        p.requires_grad_(False)

    lpips_fn = None
    if args.w_lpips > 0:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()
        except Exception as err:  # no internet for the VGG weights, say
            print(f'LPIPS unavailable ({err}); training without it', flush=True)

    student = Student(width=args.width, blocks=args.blocks).to(dev)
    ema = Student(width=args.width, blocks=args.blocks).to(dev).eval()
    ema.load_state_dict(student.state_dict())
    print(f'student: {gflops(Student(args.width, args.blocks)):.1f} GFLOPs, '
          f'{sum(p.numel() for p in student.parameters()) / 1e6:.1f} M params', flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 2000) * (0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))) * 0.95 + 0.05)
    )

    step, best = 0, -1e9
    last = os.path.join(args.out, 'ckpt_last.pt')
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        student.load_state_dict(ck['student'])
        ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        step, best = ck['step'], ck.get('best', best)
        print(f'resumed at step {step}', flush=True)

    def batch(idx_pool, n, aug):
        t = idx_pool[torch.randint(len(idx_pool), (n,))]
        s = idx_pool[torch.randint(len(idx_pool), (n,))]
        target = store[t.to(store.device)].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255
        if aug:
            target = jitter(target, args.jitter)
            flip = torch.rand(n, device=dev) < 0.5
            target = torch.where(flip[:, None, None, None], target.flip(3), target)
        src_emb = emb_all[s.to(dev)]
        latent = F.normalize(src_emb @ emap, dim=1)
        return target, latent, src_emb

    @torch.no_grad()
    def validate():
        g = torch.Generator().manual_seed(1234)
        n = args.val_pairs
        t = val_idx[torch.randint(len(val_idx), (n,), generator=g)]
        s = val_idx[torch.randint(len(val_idx), (n,), generator=g)]
        sums = dict(l1=0.0, psnr=0.0, id_teacher=0.0, id_student=0.0, id_student_vs_teacher=0.0)
        first = None
        for k in range(0, n, 32):  # in chunks, so memory stays small
            target = store[t[k:k + 32].to(store.device)].to(dev).permute(0, 3, 1, 2).float() / 255
            src_emb = emb_all[s[k:k + 32].to(dev)]
            latent = F.normalize(src_emb @ emap, dim=1)
            with torch.autocast(**amp):
                tea = teacher(target, latent).float()
                stu = ema(target, latent).float()
                e_t = F.normalize(arcface(arc_input(tea)).float(), dim=1)
                e_s = F.normalize(arcface(arc_input(stu)).float(), dim=1)
            m = len(target)
            sums['l1'] += F.l1_loss(stu, tea).item() * m
            sums['psnr'] += psnr(stu, tea) * m
            sums['id_teacher'] += F.cosine_similarity(e_t, src_emb).mean().item() * m
            sums['id_student'] += F.cosine_similarity(e_s, src_emb).mean().item() * m
            sums['id_student_vs_teacher'] += F.cosine_similarity(e_t, e_s).mean().item() * m
            if first is None:
                first = (target[:8], tea[:8], stu[:8])
        v = {k: x / n for k, x in sums.items()}
        v['id_ratio'] = v['id_student'] / max(1e-6, v['id_teacher'])
        v['step'] = step
        return (v, *first)

    def save_grid(target, tea, stu):
        import cv2

        rows = [torch.cat([a, b, c], 2) for a, b, c in zip(target, tea, stu)]  # target | teacher | student
        grid = (torch.cat(rows, 1).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, 'samples', f'step_{step:06d}.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    def save(path, extra=None):
        torch.save(dict(student=student.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step, best=best, width=args.width, blocks=args.blocks,
                        **(extra or {})), path + '.tmp')
        os.replace(path + '.tmp', path)

    log_path = os.path.join(args.out, 'train_log.csv')
    new_log = not os.path.exists(log_path)
    log = open(log_path, 'a', newline='')
    writer = csv.writer(log)
    if new_log:
        writer.writerow(['step', 'loss', 'l1', 'lpips', 'id', 'lr', 'it_per_s', 'hours'])
    started = time.time()
    tick = time.time()
    while step < args.steps:
        student.train()
        target, latent, src_emb = batch(train_idx, args.batch, aug=True)
        with torch.no_grad(), torch.autocast(**amp):
            tea = teacher(target, latent).float()
        with torch.autocast(**amp):
            stu = student(target, latent).float()
            l1 = F.l1_loss(stu, tea)
            lp = lpips_fn(stu * 2 - 1, tea * 2 - 1).mean() if lpips_fn is not None else torch.zeros((), device=dev)
            emb = F.normalize(arcface(arc_input(stu)).float(), dim=1)
            idl = (1 - F.cosine_similarity(emb, src_emb)).mean()
            loss = args.w_l1 * l1 + args.w_lpips * lp + args.w_id * idl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, ps in zip(ema.parameters(), student.parameters()):
                pe.mul_(d).add_(ps.detach(), alpha=1 - d)
        step += 1

        if step % 100 == 0:
            rate = 100 / (time.time() - tick)
            tick = time.time()
            hours = (time.time() - started) / 3600
            writer.writerow([step, f'{loss.item():.4f}', f'{l1.item():.4f}', f'{lp.item():.4f}', f'{idl.item():.4f}',
                             f'{sched.get_last_lr()[0]:.2e}', f'{rate:.2f}', f'{hours:.2f}'])
            log.flush()
            print(f'step {step} loss {loss.item():.4f} l1 {l1.item():.4f} lpips {lp.item():.4f} id {idl.item():.4f} '
                  f'{rate:.1f} it/s', flush=True)
        if step % args.val_every == 0 or step == args.steps:
            v, t8, te8, st8 = validate()
            v['hours'] = (time.time() - started) / 3600
            with open(os.path.join(args.out, 'val.jsonl'), 'a') as f:
                f.write(json.dumps(v) + '\n')
            save_grid(t8, te8, st8)
            print('VAL', json.dumps(v), flush=True)
            score = v['psnr'] / 40 + v['id_ratio']  # both matter: look like the teacher, keep the identity
            if score > best:
                best = score
                save(os.path.join(args.out, 'ckpt_best.pt'), dict(val=v))
        if step % args.ckpt_every == 0:
            save(last)
        if args.hours and time.time() - started > args.hours * 3600:
            print('time limit reached', flush=True)
            break
    save(last)
    log.close()
    print('finished at step', step, flush=True)


if __name__ == '__main__':
    main()

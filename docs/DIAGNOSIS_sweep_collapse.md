# Diagnosis: episode sweep (ep0/10/100) — 0-failure, collapse, loss @2% fg

Sumber: run Colab `[sweep] episodes=0 10 100 epoch=10 bsz=24`, backbone resnet50,
size 512, lr 1e-4, amp bf16, auxloss 0.4.

## 1. `episodes=10` tidak training sama sekali

```
# train episodes/epoch: 10 (0 batches) | val episodes: 92
Avg L: 0.00000   (semua metrik latih 0.00)
```

`build_loader` memakai `drop_last=(split=='train')`; `10 // 24 = 0` batch →
DataLoader kosong → `optimizer.step()` tidak pernah jalan → `loss_buf` kosong →
`_avg_loss()=0.0`. Blok val identik bit-per-bit 10 epoch (`3.02877 / 1.31 / 0.59 /
0.95`) = evaluasi **model random-init**. Val-nya "terbalik" (`IoU(fg)=1.31 > IoU(bg)=0.59`)
karena bobot acak memprediksi hampir semuanya foreground.

`logs/ep10_IFCNet.log/best_model.pt` = bobot acak.

## 2. Kenapa ep0 paling lama

| run | episodes | batch/epoch | kerja |
|---|---|---|---|
| ep0 | 455 | 18 | 18 |
| ep100 | 100 | 4 | 4 |
| ep10 | 10 | 0 | 0 |

18 : 4 : 0. Bukan efek strategi episode. Bonus: `drop_last` membuang 23 episode
(5%) tiap epoch di ep0.

## 3. Kolaps ke all-background (penyebab "tidak tertolong")

Val `IoU(landslide)=0.00` di semua epoch. `IoU(bg)=98.69`, `FB-IoU=97.41`,
`Dice(bg)=99.34` adalah solusi trivial prediksi background:

- pred all-bg ⇒ `inter_fg=0` ⇒ `IoU(fg)=0`
- `IoU(bg)=fraksi bg val ≈ 98.7`
- `FB-IoU = bg_area/(bg+fg) ≈ 97.4`

Model tidak pernah keluar dari fixed point itu. `best_model` disimpan di epoch 0
(val IoU 0.00 > -inf) lalu tak pernah diperbarui (semua epoch 0.00).

## 4. Loss `structure_loss` pada fg ~2%

- `weit = 1 + 5·|avgpool(mask,31) − mask|` → bobot total ≈ bg `0.98N` + fg `0.02N·6`
  = `1.10N` ⇒ **BCE masih ~89% digerakkan background** (share fg ~11%).
- Di init (σ=0.5 kedua kanal), gradien pada logit fg: `0.891·(+0.5) + 0.109·(−0.5)
  = +0.391` → menurunkan logit fg ⇒ dorongan ke all-bg.
- Suku weighted-IoU per-gambar: `union−inter = Σ_fg weit + Σ_bg pred·weit` memuat
  seluruh 98% background ⇒ gradien per piksel ~O(1/N) ≈ 1e-3, kalah ~50× dari BCE.
- `use_original_imgsize=False` meng-upsample ketiga skala ke 512 **sebelum** loss →
  skala coarse 16×16 pun terdilusi sama.

`structure_loss` dirancang untuk COD/medical (fg 5–30%); di 2% ia rusak. Metrik
headline (FB-IoU 97) juga memberi hadiah pada prediksi all-bg.

Bukti lokal (sebelum eksekusi dihentikan, CPU, data sintetis fg=kotak merah jelas):
- lr=1e-4, 24 episode × 8 epoch → val IoU(fg) = 0.00 (loss 2.60→2.21).
- overfit 1 episode, lr=1e-3, 30 step → IoU(fg) = 91.77.

⇒ Arsitektur + dataset + loss berfungsi; yang kurang: (a) step/lr dan (b) dorongan
fg pada loss.

## 5. Perbaikan yang diterapkan

| # | Perbaikan | File |
|---|---|---|
| 1 | `drop_last` hanya jika `len(dataset) >= bsz`; `episodes` 0<e<bsz → `ValueError`; loader 0 batch → `RuntimeError`; cetak jumlah episode yang dibuang | `train.py`, `sweep_episodes.sh` (fail-fast) |
| 2 | lr default `1e-4` → `3e-4` (backbone beku butuh lr lebih tinggi) | `train.py`, `train.sh` |
| 3 | Loss fg-aware: `--loss {tversky,dice,focal,bce,structure}` (default `tversky`, FN lebih dihukum: α=0.3 β=0.7) + `--pos-weight` untuk BCE; loss dihitung **per skala di resolusi asli** (gt di-downsample, bukan logit di-upsample) | `model/losses.py`, `model/mymodel.py` |
| 4 | Metrik: tambah `Prec(landslide)`/`Rec(landslide)` (recall 0 = kolaps); `[WARN]` keras bila val IoU(fg) ≤ 0 | `common/metrics.py`, `common/evaluation.py`, `train.py`, `test.py` |
| 5 | `--device auto\|cuda:0\|cpu` (bisa smoke-test CPU); `model.module` → `unwrap()`; `F.upsample` → `F.interpolate`; `Logger.log_params` hitung per `requires_grad` (SAM2 trunk beku tak lagi dilaporkan "learnable") | `train.py`, `test.py`, `model/mymodel.py`, `common/logger.py` |

`forward` kini mengembalikan logit **native** (`use_original_imgsize=True` yang
dipass train/test); upsample hanya untuk metrik final (`IFCNet.predict_mask`).

## 6. Rekomendasi run

```bash
# smoke cepat (2 epoch)
EPOCH=2 BSZ=8 LOGPATH=fix_smoke bash 1-shot/train.sh

# serius
LR=3e-4 EPOCH=100 BSZ=16 LOSS=tversky POS_WEIGHT=0 bash 1-shot/train.sh
```

Cek di log: baris `Rec(landslide)` harus > 0 sejak beberapa epoch awal. Kalau
masih 0 setelah ~20 epoch → naikkan `LR` (1e-3) atau `EPOCH`.

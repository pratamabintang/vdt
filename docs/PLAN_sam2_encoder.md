# EPIC: Swap IFCNet encoder → SAM2 Hiera (encoder-only)

**Goal** — ganti encoder `IFCNet` (torchvision resnet50) dengan **SAM2 Hiera trunk** + `Adapter` (parameter-efficient), tanpa mengubah fusion/decoder (BiCorr_v1), dataset, atau loss. 2-modal (RGB + DTM) tetap.

**Scope** — hanya encoder. **Non-goal** — ganti decoder/fusion, ubah loss/metrik/episode, sentuh `5-shot/`.

**Baseline** — tag `v1.0-landslide-rgb-dtm` (`e51849d`). Kerja di branch `encoder-swap`.

**Default varian** — `sam2_hiera_l` (`sam2_hiera_l.yaml`, `multilevel_ch = [288,576,1152]`).

**Definition of Done (epic)**
- [ ] `--backbone sam2_hiera_l` (atau varian lain) jalan end-to-end di Colab.
- [ ] Adapter menerima grad; trunk tetap beku.
- [ ] `train.sh` smoke 2 epoch tanpa OOM; loss turun; `best_model.pt` tersimpan.
- [ ] `test.sh` jalan; cetak IoU(landslide)/FB-IoU/Dice.
- [x] `py_compile` semua lolos.
- [ ] Bandingkan vs baseline; dokumentasi hasil.

**Interface contract (encoder baru)**
- `forward(x) -> [f_fine, f_mid, f_coarse]` (3 map), dari Hiera `x2,x3,x4` (stride 8/16/32). `x1` dibuang.
- `multilevel_ch = [C_fine, C_mid, C_coarse]`.
- `nbottlenecks = [1,1,1,1]` (post-trim `[1,1,1]`).

**Channel per varian** — Hiera stage `x1,x2,x3,x4` = `(stride 4,8,16,32)`. Kita pakai **x2,x3,x4**.

| varian | embed dims `x1,x2,x3,x4` | `multilevel_ch = [x2,x3,x4]` |
|---|---|---|
| `sam2_hiera_t` | 96,192,384,768 | `[192,384,768]` |
| `sam2_hiera_s` | 96,192,384,768 | `[192,384,768]` |
| `sam2_hiera_b+` | 112,224,448,896 | `[224,448,896]` |
| `sam2_hiera_l` | 144,288,576,1152 | `[288,576,1152]` |

> Dikonfirmasi ulang lewat probe (SAM2-4/SAM2-8); sesuaikan bila pakai varian 2.1.

---

## Board

| ID | Judul | Label | Prio | Depends | Est |
|---|---|---|---|---|---|
| SAM2-1 | Vendor SAM2 (kode + config + checkpoint) | infra | P0 | — | S |
| SAM2-2 | `Adapter` + `SAM2Encoder` | encoder | P0 | SAM2-1 | M |
| SAM2-3 | `extract_feat_sam2` (feature.py) | encoder | P0 | SAM2-2 | S |
| SAM2-4 | Integrasi `IFCNet` (branch sam2) | encoder | P0 | SAM2-3 | M |
| SAM2-5 | Fix `torch.no_grad` mematikan adapter | bug | P0 | SAM2-4 | S |
| SAM2-6 | Plumbing arg train/test + sh | config | P0 | SAM2-4 | S |
| SAM2-7 | `requirements.txt` (hydra/omegaconf) | infra | P0 | SAM2-1 | XS |
| SAM2-8 | Probe bentuk + grad + py_compile | test | P0 | SAM2-6 | S |
| SAM2-9 | Smoke train 2 epoch (Colab, hiera_l) | test | P1 | SAM2-8 | M |
| SAM2-10 | Test + banding vs baseline | test | P1 | SAM2-9 | M |
| SAM2-11 | Dokumentasi hasil & keputusan | docs | P2 | SAM2-10 | S |

Prioritas: P0 = blocker, P1 = penting, P2 = nice. Est: XS/S/M/L.

---

## Status (implementasi lokal)

| ID | Status | Catatan |
|---|---|---|
| SAM2-1 | ✅ done | `third_party/sam2/` (vendored, dipangkas ke paket `sam2/` 0.45 MB) |
| SAM2-2 | ✅ done | `1-shot/model/sam2_encoder.py` |
| SAM2-3 | ✅ done | `extract_feat_sam2` di `base/feature.py` |
| SAM2-4 | ✅ done | branch `sam2_hiera_*` di `mymodel.py` |
| SAM2-5 | ✅ done | flag `frozen_grad_backbone` |
| SAM2-6 | ✅ done | `--sam2-config/--sam2-ckpt/--sam2-adapters` + sh |
| SAM2-7 | ✅ done | `hydra-core`, `omegaconf`, `iopath` |
| SAM2-8 | ✅ done | probe lokal (CPU) lulus |
| SAM2-9 | ⏳ Colab | butuh GPU + checkpoint |
| SAM2-10 | ⏳ | — |
| SAM2-11 | ⏳ | — |

**Deviasi dari rencana awal (lebih baik)**

1. **Trunk-only build**, bukan `build_sam2()` + `del` decoder. `SAM2Encoder` cuma
   instantiate node `model.image_encoder.trunk` dari config Hydra, lalu memuat
   bobot `image_encoder.trunk.*` dari checkpoint. Jadi prompt-encoder, mask-decoder,
   memory dan FPN-neck **tidak pernah dibangun** (lebih hemat memori, tanpa
   `del`/try-except per atribut).
2. **Config name** SAM2-UNet (`sam2_hiera_l.yaml`) ternyata sebuah *shim* (isinya
   path), bukan config Hydra asli → `_compose_trunk_cfg()` mencoba
   `sam2_hiera_l.yaml` dulu lalu otomatis jatuh ke `configs/sam2/sam2_hiera_l.yaml`.
3. **Dep tambahan**: `iopath` (di-import `hieradet.py`), selain `hydra-core`/`omegaconf`.
4. **Grad path** dikontrol flag eksplisit `frozen_grad_backbone` (bukan memeriksa
   `self.training`), sehingga `backbone.eval()` yang dipanggil `IFCNet.train_mode()`
   tidak mematikan grad adapter.
5. **Vendoring** dipangkas ke paket `sam2/` saja (assets/notebooks/training/demo
   dibuang) + `VENDOR.md`; `pip install sam2` tidak dipakai.

**Hasil probe lokal (CPU, input 128)**

```
hiera_t: out_channels=(192,384,768)  feats [1,192,16,16] [1,384,8,8] [1,768,4,4]
hiera_l: out_channels=(288,576,1152) feats [1,288,16,16] [1,576,8,8] [1,1152,4,4]
         adapter params 1,714,416 | trunk params 212,149,296 (frozen)
         adapter grad=True  trunk grad=False  decoder grad=True
IFCNet + SAM2: 3 logits (2,2,128,128), loss OK, grad mengalir ke adapter + decoder
```

---

## SAM2-1 — Vendor SAM2 (kode + config + checkpoint)
**Label** infra · **P0** · **Depends** —

**Desc**
- Clone `https://github.com/facebookresearch/sam2` ke `third_party/sam2/`, lalu pangkas ke paket `sam2/` saja (+ lisensi, `VENDOR.md`).
- Pastikan `import sam2` jalan (tambah `sys.path` di `sam2_encoder.py`).
- Unduh checkpoint `sam2_hiera_large.pt` (**versi 2.0, bukan 2.1**).

**Acceptance**
- [x] `python -c "import sam2; from hydra import compose; print('ok')"` jalan.
- [x] `_compose_trunk_cfg('sam2_hiera_l.yaml')` menemukan node trunk (embed_dim 144, stages [2,6,36,4]).
- [ ] File `.pt` ada dan path dikonfig (`SAM2_CKPT`).

**Notes/risk** — konfig 2.1 beda nama; jangan campur. Nama config SAM2-UNet (`sam2_hiera_l.yaml`) bukan config Hydra asli → lihat deviasi #2 di Status.

---

## SAM2-2 — `Adapter` + `SAM2Encoder`
**Label** encoder · **P0** · **Depends** SAM2-1
**File** BARU `1-shot/model/sam2_encoder.py`

**Desc**
- Salin `Adapter` dari `SAM2UNet.py` (Linear→GELU→Linear→GELU, residual, `qkv.in_features`): `prompt = MLP(x)`, `out = block(x + prompt)`.
- `class SAM2Encoder(nn.Module)` (**implementasi final = trunk-only**):
  - `_compose_trunk_cfg(config)` → `cfg.model.image_encoder.trunk` dari Hydra (fallback `sam2_hiera_*.yaml` → `configs/sam2/...`).
  - `instantiate(trunk_cfg)`; muat `image_encoder.trunk.*` dari checkpoint (kalau ada). Decoder/neck **tidak dibangun**.
  - Freeze: `for p in trunk.parameters(): p.requires_grad = False`.
  - `trunk.blocks = nn.ModuleList([Adapter(b) for b in trunk.blocks])` (jika `adapters=True`).
  - `out_channels` = `trunk.channel_list[2], [1], [0]` = `(c2,c3,c4)`, `strides = (8,16,32)`.
  - `forward(x) -> [x2,x3,x4]` (buang `x1` dari 4 output trunk).
  - `train()` → paksa `trunk.eval()` (DropPath/pos-embed); `requires_grad_features()` = `use_adapters`.

**Acceptance**
- [x] Modul ter-import; `forward` return 3 tensor.
- [x] Adapter punya grad; `trunk` semua `requires_grad=False` & tanpa grad.

---

## SAM2-3 — `extract_feat_sam2`
**Label** encoder · **P0** · **Depends** SAM2-2
**File** `1-shot/model/base/feature.py`

**Desc**
- Tambah `def extract_feat_sam2(img, encoder, feat_ids=None, bottleneck_ids=None, lids=None): return encoder(img)` (signature kompatibel `extract_feats`).

**Acceptance**
- [ ] `extract_feat_sam2(x, enc)` panjang 3, cocok `encoder.out_channels`.

---

## SAM2-4 — Integrasi `IFCNet`
**Label** encoder · **P0** · **Depends** SAM2-3
**File** `1-shot/model/mymodel.py`

**Desc**
- Tambah `backbone in {sam2_hiera_t, sam2_hiera_s, sam2_hiera_b+, sam2_hiera_l}`:
  - `self.backbone = SAM2Encoder(cfg, ckpt, adapters=...)`.
  - `self.extract_feats = extract_feat_sam2`.
  - `self.multilevel_ch = [c2,c3,c4]`; `self.nbottlenecks = [1,1,1,1]`.
- `forward` signature tetap (`query_img, query_img_th, support_img, support_img_th, support_mask`).
- Pastikan `layer_ids` → `[0,1,2,3]` → 3 grup; `multi_smi_fuse` post-trim `[1,1,1]`.

**Acceptance**
- [ ] `IFCNet('sam2_hiera_b+', False)` konstruk tanpa error.
- [ ] `forward` 5-arg → `logit_masks` list 3 skala, diinterpolate ke 512.

**Blocked-risk** — keberadaan `image_encoder.neck`/atribut lain antar versi.

---

## SAM2-5 — Fix `torch.no_grad` mematikan adapter
**Label** bug · **P0** · **Depends** SAM2-4
**File** `1-shot/model/mymodel.py`

**Desc**
- `forward` membungkus ekstraksi fitur dengan `torch.no_grad()`. Untuk SAM2, adapter butuh grad → non-aktifkan `no_grad` untuk encoder SAM2 (andalkan `requires_grad=False` pada trunk). Untuk resnet tetap boleh `no_grad` (tak ada param trainable).
- Rancang flag, mis. `self.trainable_encoder = True` untuk SAM2.

**Acceptance**
- [ ] Backward 1 langkah: `adapter` param `.grad` tidak None; `trunk` param `.grad` None.
- [ ] Baseline resnet tak berubah perilaku.

---

## SAM2-6 — Plumbing arg train/test + sh
**Label** config · **P0** · **Depends** SAM2-4
**File** `1-shot/train.py`, `1-shot/test.py`, `1-shot/train.sh`, `1-shot/test.sh`

**Desc**
- Arg baru: `--sam2-config` (default `sam2_hiera_b+.yaml`), `--sam2-ckpt` (path `.pt`), `--sam2-adapters` (default 1).
- Teruskan ke `IFCNet(...)` (tambah param opsional).
- `train.sh`/`test.sh`: var `BACKBONE`, `SAM2_CONFIG`, `SAM2_CKPT`, `SAM2_ADAPTERS`; `--backbone` tetap.

**Acceptance**
- [ ] `bash train.sh` dengan `BACKBONE=sam2_hiera_b+` meneruskan arg benar (cek log args).

---

## SAM2-7 — `requirements.txt`
**Label** infra · **P0** · **Depends** SAM2-1
**File** `requirements.txt`

**Desc**
- Tambah `hydra-core`, `omegaconf` (+ `iopath` bila import SAM2 menuntut).
- **Tidak** `pip install sam2` (vendored).

**Acceptance**
- [ ] `pip install -r requirements.txt` di env bersih → `build_sam2` import ok.

---

## SAM2-8 — Probe bentuk + grad + py_compile
**Label** test · **P0** · **Depends** SAM2-6

**Desc**
- Skrip probe: input `(1,3,512,512)` → cek `len==3`, shapes `[B,C,512/8..512/32]`, channels sesuai varian, grad adapter vs trunk.
- `python -m py_compile` semua `.py`.

**Acceptance**
- [ ] Shapes: b+ → `[B,224,64,64] [B,448,32,32] [B,896,16,16]` (input 512).
- [ ] Grad check (SAM2-5) lulus.
- [ ] `py_compile` exit 0.

---

## SAM2-9 — Smoke train 2 epoch (Colab, hiera_b+)
**Label** test · **P1** · **Depends** SAM2-8

**Desc**
- `EPOCH=2 BSZ=2 BACKBONE=sam2_hiera_b+ SAM2_CKPT=... bash 1-shot/train.sh` (AMP bf16, L4).
- Catat VRAM & waktu/epoch.

**Acceptance**
- [ ] Tak OOM; loss & IoU ter-log per epoch.
- [ ] `best_model.pt` ditulis.

**Notes** — jika OOM: turunkan `BSZ`, coba `hiera_s`, atau gradient checkpointing (tiket baru bila perlu).

---

## SAM2-10 — Test + banding vs baseline
**Label** test · **P1** · **Depends** SAM2-9

**Desc**
- Jalankan training penuh (epoch sama dgn baseline) → `test.sh` → catat IoU(landslide)/FB-IoU/Dice.
- Bandingkan dgn `v1.0-landslide-rgb-dtm` pada split/protokol identik.

**Acceptance**
- [ ] Tabel: baseline vs SAM2 (backbone, param, epoch, IoU/FB-IoU/Dice, waktu).
- [ ] Keputusan: lanjut/rollback.

---

## SAM2-11 — Dokumentasi hasil & keputusan
**Label** docs · **P2** · **Depends** SAM2-10
**File** `docs/` (BARU)

**Desc**
- Catat konfiguration final, varian terbaik, catatan memori, dan apakah encoder resnet tetap disediakan sebagai opsi.

**Acceptance**
- [ ] Doc singkat + perintah reproduksi.

---

## Guardrail
- Jangan sentuh `5-shot/`.
- `train.py`/`test.py`: perubahan minimal (arg + IFCNet call).
- Setiap tiket: `py_compile` sebelum tandai done.
- Rollback: `git checkout v1.0-landslide-rgb-dtm`.

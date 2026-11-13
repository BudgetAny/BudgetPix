#!/usr/bin/env bash
# Every BudgetPix test on one GPU (about 30 minutes); stops at the first failure.
#
#   MINIT2I_ROOT=... PIXELDIT_ROOT=... PMF_ROOT=... \
#   MINIT2I_WEIGHTS=<released minit2i-b-16 dir> PIXELDIT_WEIGHTS=<released pixeldit_t2i_v1.pth> bash tests/run_gpu.sh
#
#   CKPT_DIR          the BudgetPix checkpoints (scripts/download_checkpoints.py --all, plus pmf_released_b16); default checkpoints/
#   MINIT2I_ROOT      upstream checkouts, as in the README (Installation)
#   PIXELDIT_ROOT
#   PMF_ROOT
#   MINIT2I_WEIGHTS   the released MiniT2I-B/16 (MiniT2I/MiniT2I on the Hub, its minit2i-b-16 folder): MiniT2I training test
#   PIXELDIT_WEIGHTS  the released PixelDiT (nvidia/PixelDiT-1300M-1024px, pixeldit_t2i_v1.pth): PixelDiT training test
#   FAMILIES          a subset of "jit minit2i pixeldit pmf" (default: all four)
#   REF               optional: the samples/ folder of an earlier run; this run's samples must match it pixel for pixel
#   OUT               test outputs (default runs/tests)
#
# 1. layouts (CPU)   2. sampling from every BudgetPix checkpoint family, dense and at a budget   3. comparison with REF
# 4. tiny synthetic training data   5. training: every family runs 2 steps from the released weights, saves, resumes to 3
set -eo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
CK=$(cd "${CKPT_DIR:-checkpoints}" && pwd)
OUT=$(mkdir -p "${OUT:-runs/tests}" && cd "${OUT:-runs/tests}" && pwd)
FAMILIES=${FAMILIES:-"jit minit2i pixeldit pmf"}
S=$OUT/samples D=$OUT/data
mkdir -p "$S" "$D"
step() { echo; echo "=== $*"; }
has() { [[ " $FAMILIES " == *" $1 "* ]]; }
RUN="python -m torch.distributed.run --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000))"

step "1/5 layouts: tiling, budgets, packing (CPU)"
python tests/test_layouts.py

step "2/5 sampling from the BudgetPix checkpoints"
if has jit; then (cd jit
  python sample.py --ckpt "$CK/budgetpix_jit_b16_256.pth" --budget 128 --classes 207,88 --steps 10 --overlay --out "$S/jit_b16_b128.png"
  python sample.py --ckpt "$CK/budgetpix_jit_b16_256.pth" --classes 207 --steps 10 --out "$S/jit_b16_dense.png"
  python sample.py --ckpt "$CK/budgetpix_jit_b32_512.pth" --budget 128 --classes 88 --steps 10 --overlay --out "$S/jit_b32_b128.png"); fi
if has pmf; then (cd pmf
  python sample.py --ckpt "$CK/budgetpix_pmf_l16_256.pt" --classes 207,88 --budget 128 --overlay --out "$S/pmf_b128.png"
  python sample.py --ckpt "$CK/budgetpix_pmf_l16_256.pt" --classes 207 --budget 256 --out "$S/pmf_dense.png"); fi
if has minit2i; then (cd minit2i
  python sample.py --ckpt "$CK/budgetpix_minit2i_b_512.pt" --prompts "a red fox in a snowy forest" --budget 512 --steps 10 --out "$S/minit2i_b_b512.png"); fi
if has pixeldit; then (cd pixeldit
  python sample.py --ckpt "$CK/budgetpix_pixeldit_1024.pt" --prompts "a red fox in a snowy forest" --budget 2048 --steps 8 --out "$S/pixeldit_b2048.png"); fi

step "3/5 comparison with an earlier run's samples"
if [ -n "$REF" ]; then python tests/compare_images.py "$S" "$REF"; else echo "skipped (set REF to compare)"; fi

step "4/5 tiny synthetic training data"
python tests/make_tiny_data.py --out "$D" $(has jit && echo --jit_from "$CK/budgetpix_jit_b16_256.pth")
if has pixeldit; then (cd pixeldit
  python scripts/precompute_data.py --images "$D/pixeldit_images" --out "$D/pixeldit_chunks" --size 1024 --per_chunk 8
  python scripts/encode_prompts.py --null_only --out "$D/null_emb.pt"); fi

step "5/5 training: 2 steps, a checkpoint, then --resume to step 3"
train_twice() {   # <name> <command...>: run to step 2, then resume to step 3
  local name=$1; shift
  rm -rf "$OUT/train_$name"
  "$@" 2; "$@" 3 --resume
  ls "$OUT/train_$name"/checkpoint-last.* > /dev/null
  echo "[train] $name: trained, saved and resumed"
}
jit_train()      { (cd jit && $RUN train.py --released "$D/jit_released.pth" --data "$D/imagenet" --out "$OUT/train_jit" \
                     --batch_size 2 --grad_accum 1 --num_workers 1 --save_every 1 --log_every 1 --max_steps "$@"); }
minit2i_train()  { local n=$1; shift; (cd minit2i && $RUN train.py --config configs/minit2i_b_512.yml --out "$OUT/train_minit2i" "$@" \
                     --set checkpoint="${MINIT2I_WEIGHTS:?}" data.local_dataset_dir="$D/minit2i" data.micro_batch_size=1 \
                           data.num_workers=1 batch_size=1 grad_accum=1 max_steps=$n save_every=1 log_every=1); }
pixeldit_train() { local n=$1; shift; (cd pixeldit && $RUN train.py --config configs/pixeldit_1024.yml --out "$OUT/train_pixeldit" "$@" \
                     --set checkpoint="${PIXELDIT_WEIGHTS:?}" null_emb="$D/null_emb.pt" "data.chunks=[$D/pixeldit_chunks]" \
                           data.micro_batch_size=1 data.num_workers=1 batch_size=1 grad_accum=1 max_steps=$n save_every=1 \
                           log_every=1 snapshot_every=0); }
pmf_train()      { local n=$1; shift; (cd pmf && $RUN train.py --arch B16 --ckpt "$CK/pmf_released_B16_full_uv.pt" \
                     --data "$D/imagenet" --out "$OUT/train_pmf" --batch 2 --workers 1 --save_every 1 --log_every 1 --steps $n "$@"); }
for fam in $FAMILIES; do train_twice "$fam" "${fam}_train"; done

echo; echo "ALL BUDGETPIX TESTS PASSED"

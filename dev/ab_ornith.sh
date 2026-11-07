#!/bin/bash
# Ornith 1.5 A/B sweep: env-flag variants + draft swap + prefill flags.
# Decode runs pass --ane-ffn-share 0 (GPU-only prefill; decode unaffected,
# skips the ~21 s ANE calibration). Prefill runs pin the calibrated 0.375
# share so every config sees the same split without recalibrating.
# Exit 1 = perf-gate trips (known B3 cliff), JSON is still complete.
set -u
cd "$(dirname "$0")/.."
BB=./build/engine-tests/backend-benchmark
LIB=build/engine-tests/production-and-test.metallib
OUT=build/ab-ornith
mkdir -p "$OUT"

N9=install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit
N9D2=install/models/test/Ornith-9B-DFlash2
M35=install/models/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit

run() { # name modelroot scenario samples aneShare [ENV=VAL ...]
  local name=$1 model=$2 scenario=$3 samples=$4 ane=$5; shift 5
  if [ -s "$OUT/$name.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name  $(date +%H:%M:%S)  env: $* ==="
  env "$@" "$BB" "$LIB" "$model" --samples "$samples" --scenario "$scenario" \
      --ane-ffn-share "$ane" > "$OUT/$name.json" 2> "$OUT/$name.stderr"
  local rc=$?
  echo "--- $name exit=$rc $(date +%H:%M:%S) ---"
}

D=decode
P=short
S=3
Z=0      # decode: skip ANE calibration
A=0.375  # prefill: pinned calibrated share

# ---------- Ornith-1.5-9B decode flags (V1 draft, production install) ----------
run n9-base        $N9  $D $S $Z
run n9-tree        $N9  $D $S $Z RICHENGINE_VERIFY_TREE=1
run n9-headm64     $N9  $D $S $Z RICHENGINE_HEAD_M64=1
run n9-headperlane $N9  $D $S $Z RICHENGINE_HEAD_PER_LANE=1
run n9-pool        $N9  $D $S $Z RICHENGINE_DFLASH_POOL=1
run n9-noadapt     $N9  $D $S $Z RICHENGINE_ADAPTIVE_PROPOSALS=0
run n9-nobypass    $N9  $D $S $Z RICHENGINE_DRAFT_BYPASS=0
run n9-nochain     $N9  $D $S $Z RICHENGINE_CHAIN_PREDRAFT=0
run n9-nogpupred   $N9  $D $S $Z RICHENGINE_GPU_PREDRAFT=0
run n9-cap8        $N9  $D $S $Z RICHENGINE_PROPOSAL_CAP=8
run n9-cap12       $N9  $D $S $Z RICHENGINE_PROPOSAL_CAP=12
run n9-r2off       $N9  $D $S $Z RICHENGINE_VERIFY_SPLIT_R2_OFF=1
run n9-ring2off    $N9  $D $S $Z RICHENGINE_DECODE_RING2_OFF=1
run n9-submit      $N9  $D $S $Z RICHENGINE_SUBMIT_AHEAD=1
run n9-headfusedoff $N9 $D $S $Z RICHENGINE_HEAD_FUSED_OFF=1
run n9-icboff      $N9  $D $S $Z RICHENGINE_ICB_OFF=1
run n9-nofusedgate $N9  $D $S $Z RICHENGINE_NO_FUSED_GATE=1
run n9-mixedpackoff $N9 $D $S $Z RICHENGINE_MIXED_DECODE_PACK_OFF=1
run n9-spanbankoff $N9  $D $S $Z RICHENGINE_SPAN_BANK_OFF=1
run n9-dfamode-forced $N9 $D $S $Z RICHENGINE_DFA_MODE=forced
run n9-dfamode-off $N9  $D $S $Z RICHENGINE_DFA_MODE=off

# ---------- Ornith-1.5-9B draft swap: DFlash2 graft ----------
run n9d2-base      $N9D2 $D $S $Z
run n9d2-pool      $N9D2 $D $S $Z RICHENGINE_DFLASH_POOL=1
run n9d2-noadapt   $N9D2 $D $S $Z RICHENGINE_ADAPTIVE_PROPOSALS=0
run n9d2-cap12     $N9D2 $D $S $Z RICHENGINE_PROPOSAL_CAP=12

# ---------- Ornith-1.5-9B prefill flags ----------
run n9p-base       $N9  $P 2 $A
run n9p-gdn64      $N9  $P 2 $A RICHENGINE_GDN_CHUNKED=64
run n9p-gdn128     $N9  $P 2 $A RICHENGINE_GDN_CHUNKED=128
run n9p-ksplit4    $N9  $P 2 $A RICHENGINE_PREFILL_Q4_KSPLIT=4
run n9p-persist8   $N9  $P 2 $A RICHENGINE_PREFILL_Q4_PERSISTENT=8
run n9p-m64        $N9  $P 2 $A RICHENGINE_PREFILL_Q4_M64=1
run n9p-int8       $N9  $P 2 $A RICHENGINE_PREFILL_FAST_INT8=1
run n9p-splitsmax8 $N9  $P 2 $A RICHENGINE_PREFILL_SPLITS_MAX=8

# ---------- Ornith-1.5-35B-A3B decode flags (DFlash2 graft v2) ----------
run m35-base       $M35 $D $S $Z
run m35-sharedens  $M35 $D $S $Z RICHENGINE_MOE_SHARED_DENSE=1
run m35-decodem32  $M35 $D $S $Z RICHENGINE_MOE_DECODE_M32=1
run m35-sg4        $M35 $D $S $Z RICHENGINE_MOE_SG4=1
run m35-noverrows  $M35 $D $S $Z RICHENGINE_MOE_VERIFIED_ROWS_OFF=1
run m35-cap8       $M35 $D $S $Z RICHENGINE_PROPOSAL_CAP=8
run m35-cap16      $M35 $D $S $Z RICHENGINE_PROPOSAL_CAP=16
run m35-tree       $M35 $D $S $Z RICHENGINE_VERIFY_TREE=1
run m35-headm64    $M35 $D $S $Z RICHENGINE_HEAD_M64=1
run m35-headperlane $M35 $D $S $Z RICHENGINE_HEAD_PER_LANE=1
run m35-pool       $M35 $D $S $Z RICHENGINE_DFLASH_POOL=1
run m35-noadapt    $M35 $D $S $Z RICHENGINE_ADAPTIVE_PROPOSALS=0
run m35-icboff     $M35 $D $S $Z RICHENGINE_ICB_OFF=1
run m35-packedoff  $M35 $D $S $Z RICHENGINE_MOE_PACKED_OFF=1
run m35-union64    $M35 $D $S $Z RICHENGINE_MOE_UNION=64

echo "ALL DONE $(date +%H:%M:%S)"

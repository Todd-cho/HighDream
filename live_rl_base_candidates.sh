#!/bin/bash
# RL1~RL13 base-checkpoint live candidates -- raw RL only (no w/p acq layer),
# safety override on, new run/rl naming convention (2026-08-21).
# RL1-RL3 (v2/v3/v4) trained on the OLD 4-angle pool [0,8,15,22deg], NOT
# [0,90] -- a 91deg live start is out-of-distribution for them, expect worse.
#
# Usage: ./live_rl_base_candidates.sh <N>   (N = 1..13)
# Run ONE at a time -- each needs the live server ready + you watching.

set -e
N="$1"
if [ -z "$N" ]; then
  echo "Usage: $0 <rl-number 1-13>"
  echo "  1=v2_300iter          2=v3_300iter           3=v4_300iter"
  echo "  4=v5_angle090_50iter  5=v6_angle090_50iter   6=v6b_angle090_aggro_50iter"
  echo "  7=v7_angle090_hardopp_50iter  8=v8_angle090_opp055_50iter (shared base)"
  echo "  9=v9_leadpursuit040_50iter (ALREADY DONE, run22, failed live)"
  echo "  10=v10_ratelimit005_50iter  11=v11a_precision10_20iter"
  echo "  12=v11b_precision20_20iter  13=v12_precision10_50iter (best JSBSim)"
  exit 1
fi

declare -A TAGS=(
  [1]=v2_300iter [2]=v3_300iter [3]=v4_300iter
  [4]=v5_angle090_50iter [5]=v6_angle090_50iter [6]=v6b_angle090_aggro_50iter
  [7]=v7_angle090_hardopp_50iter [8]=v8_angle090_opp055_50iter
  [9]=v9_leadpursuit040_50iter [10]=v10_ratelimit005_50iter
  [11]=v11a_precision10_20iter [12]=v11b_precision20_20iter
  [13]=v12_precision10_50iter
)
declare -A RUNNUM=(
  [1]=0023 [2]=0024 [3]=0025 [4]=0026 [5]=0027 [6]=0028 [7]=0029 [8]=0030
  [9]=0022 [10]=0031 [11]=0032 [12]=0033 [13]=0034
)

TAG="${TAGS[$N]}"
RUNNUM="${RUNNUM[$N]}"
if [ -z "$TAG" ]; then
  echo "no such rl number: $N"
  exit 1
fi

BASE="artifacts/models/highdream/altitude_attack_followup_v1_stage6obs19_${TAG}_C10"
LOG="artifacts/logs"

echo "[rl${N}] tag=${TAG} run=${RUNNUM}"
python run_unreal_inference.py --mode rl \
  --bundle-dir "$BASE" \
  --observation-mode tactical19 --team-name blue \
  --server-ip 127.0.0.1 --server-port 9999 --action-repeat 6 \
  --safety-override \
  --log-csv "${LOG}/run${RUNNUM}_raw_rl${N}_saf1_ang091_seed01.csv" \
  --damage-log-csv "${LOG}/run${RUNNUM}_damage_raw.csv"

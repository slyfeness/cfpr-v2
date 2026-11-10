#!/usr/bin/env bash
set -euo pipefail
root=/data/lanyi/outputs/cfpr-v2/identity_injection_pilot/run_20261110
out="$root/public_previews"
ffmpeg=/data/lanyi/.conda/envs/cfpr-v2-acc/bin/ffmpeg
mkdir -p "$out"
for sample in CRB100_001 CRB100_025; do
  for variant in P0 P1 P4 P5 P7; do
    "$ffmpeg" -hide_banner -loglevel error -y \
      -i "$root/$sample/clean_a/preview.mp4" -i "$root/$sample/$variant/preview.mp4" \
      -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' -map 0:a? \
      -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest \
      "$out/${sample}_${variant}_vs_clean.mp4"
  done
done

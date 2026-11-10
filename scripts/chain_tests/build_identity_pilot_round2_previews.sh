#!/usr/bin/env bash
set -euo pipefail

# Reads completed pilot videos and makes the six owner-approved, audio-bearing
# side-by-side review previews.  No frames are generated or altered here.
first=/data/lanyi/outputs/cfpr-v2/identity_injection_pilot/run_20261110
second=/data/lanyi/outputs/cfpr-v2/identity_injection_pilot/round2_20261110
out="$second/public_previews"
ffmpeg=/data/lanyi/.conda/envs/cfpr-v2-acc/bin/ffmpeg
mkdir -p "$out"
for sample in CRB100_001 CRB100_025; do
  for variant in P1b D75 H50; do
    "$ffmpeg" -hide_banner -loglevel error -y \
      -i "$first/$sample/clean_a/preview.mp4" -i "$second/$sample/$variant/preview.mp4" \
      -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' -map 0:a? \
      -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest \
      "$out/${sample}_${variant}_vs_clean.mp4"
  done
done

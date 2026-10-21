#!/bin/sh
# Rebuild models/final/: every configuration with >= 5 CV runs, trained once on all 229 patients
# (scripts/train_final.py). Frozen-feature models (CPU) and fine-tuned CNNs (MPS) run in parallel;
# existing files are skipped, so the script can be resumed. ~45 min frozen, ~1.5 h fine-tuned on an M1.
#   sh scripts/train_all_final.sh
#   .venv/bin/python scripts/watch_training.py      # live view
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs models/final

train() {  # variant encoder
  out=models/final/$1_$2.pt
  [ -e "$out" ] && return
  echo "=== $1 $2"
  .venv/bin/python scripts/train_final.py --variant "$1" --encoder "$2" --out "$out" 2>&1 | grep -E "^saved|Error|need the CV" | cut -c1-160
}

frozen() {
  for e in efficientnet_b0 densenet121 resnet50 convnext_tiny swin_tiny dinov2_small; do
    for v in IMG A0 B4 A1 A2 A3 A4 A5 A6 X1 X2 X3 X4 M2 M3 M5 M6 N1 N2 N3 N4 N5 N6 C1 C2 C3; do
      [ "$(ls outputs/runs/${v}__${e}_224__cv*__s*.json 2>/dev/null | wc -l)" -lt 5 ] && continue
      [ "$v/$e" = "A6/efficientnet_b0" ] && continue   # models/coremil_final.pt
      [ "$v/$e" = "IMG/densenet121" ] && continue      # models/cnn_densenet121_final.pt
      train "$v" "$e"
    done
  done
  echo FROZEN_DONE
}

finetuned() {  # FT_A6 is models/coremil_ft_final.pt
  for ve in FT_IMG:densenet121 FT2_IMG:efficientnet_b0 FT2_IMG:densenet121 FT2_IMG:resnet50; do
    train "${ve%%:*}" "${ve##*:}"
  done
  echo FT_DONE
}

frozen > logs/final_all_frozen.out 2>&1 &
finetuned > logs/final_all_ft.out 2>&1 &
wait
grep -h -E "^saved|Error|need the CV" logs/final_all_frozen.out logs/final_all_ft.out | grep -vc "^saved" | xargs echo "problems:"
ls models/final | wc -l | xargs echo "models in models/final:"

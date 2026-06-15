```bash
for items in high_diversity.json low_entropy.json high_entropy.json null; do
for items in high_diversity.json; do
  for permute_mode in in-tile in-batch; do
        for run_id in 40utmvmw j5x4suw2 owzcohia fif5wuld; do
            echo "${items} with ${permute_mode} on ${run_id}"
            uv run python scripts/baselines/early-fusion-permuted.py \
              --config configs/baselines/early-fusion-permuted.yaml \
              --run_id $run_id \
              --items_path $items \
              --permute_mode $permute_mode
        done
    done
done
```

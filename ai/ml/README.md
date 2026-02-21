# PyTorch workspace

Place PyTorch training, evaluation, and distillation scripts here.

## Tabular training module

`train.py` provides a reusable train/test pipeline for `.csv`, `.xls`, and `.xlsx` files.

Programmatic usage:

```python
from ai.pytorch.train import TrainingConfig, train_model_from_file, predict_rows

cfg = TrainingConfig(target_column="quality", task="auto", epochs=200)
bundle, metrics = train_model_from_file("path/to/dataset.csv", cfg)
predictions = predict_rows(bundle, [{"fixed acidity": 7.4, "alcohol": 9.4}])
```

CLI usage:

```bash
python ai/pytorch/train.py \
  --data ai/python/sample_data/wine+quality/winequality-red.csv \
  --target quality \
  --task auto \
  --epochs 200 \
  --save-dir ai/pytorch/artifacts/wine-red
```

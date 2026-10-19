# Fish Disease Prototype

`fish_disease.onnx` is a two-class MobileNetV3-Small baseline fine-tuned with 30 fish images from the BD Fish & Shrimp Disease Dataset. `fish_presence.onnx` is a pretrained ImageNet fish-category gate that rejects likely non-fish photos before classification. See `metrics.json` for the split and prototype metrics; six test examples are far too few to establish useful accuracy.

The dataset is licensed CC BY-NC-SA 4.0. Preserve attribution and non-commercial/share-alike restrictions when using or redistributing this model.

The separate `fish_7class/` output directory is created by `ml/notebooks/disease/train_fish_disease_7class.py`. It contains a seven-class model trained on the full fish training split and evaluation reports; the current API continues to use the two-class model files above.

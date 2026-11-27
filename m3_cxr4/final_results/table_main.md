| Model | Family | Split | Input | Ablation | Seeds | Accuracy | Macro-F1 | Macro-AUROC | ECE → calibrated |
|---|---|---|---|---|---|---|---|---|---|
| swin_t | Transformer | image | none | none | 3 | 0.9599 ± 0.0015 | 0.9665 ± 0.0014 | 0.9962 ± 0.0003 | 0.019 → 0.005 |
| efficientnet_b0 | CNN | image | none | none | 3 | 0.9566 ± 0.0017 | 0.9655 ± 0.0012 | 0.9952 ± 0.0001 | 0.020 → 0.007 |
| ensemble | Ensemble | group | none | none | 6 | 0.9583 ± 0.0019 | 0.9649 ± 0.0020 | 0.9958 ± 0.0003 | 0.011 → 0.011 |
| resnet50 | CNN | image | none | none | 3 | 0.9557 ± 0.0016 | 0.9640 ± 0.0010 | 0.9956 ± 0.0003 | 0.021 → 0.007 |
| convnext_tiny | CNN | group | none | none | 3 | 0.9573 ± 0.0016 | 0.9633 ± 0.0017 | 0.9950 ± 0.0002 | 0.031 → 0.010 |
| efficientnet_b0 | CNN | group | none | no_augment | 2 | 0.9544 ± 0.0005 | 0.9627 ± 0.0001 | 0.9925 ± 0.0013 | 0.036 → 0.008 |
| swin_t | Transformer | group | none | none | 3 | 0.9564 ± 0.0009 | 0.9626 ± 0.0013 | 0.9950 ± 0.0002 | 0.025 → 0.009 |
| efficientnet_b0 | CNN | group | none | no_class_weights | 2 | 0.9536 ± 0.0015 | 0.9624 ± 0.0016 | 0.9946 ± 0.0006 | 0.028 → 0.009 |
| efficientnet_b0 | CNN | group | none | no_clahe | 2 | 0.9536 ± 0.0029 | 0.9622 ± 0.0027 | 0.9943 ± 0.0007 | 0.025 → 0.009 |
| efficientnet_b0 | CNN | group | none | none | 3 | 0.9537 ± 0.0013 | 0.9614 ± 0.0010 | 0.9946 ± 0.0003 | 0.025 → 0.009 |
| densenet121 | CNN | group | none | none | 3 | 0.9539 ± 0.0018 | 0.9601 ± 0.0013 | 0.9945 ± 0.0004 | 0.026 → 0.006 |
| resnet50 | CNN | group | none | none | 3 | 0.9531 ± 0.0024 | 0.9596 ± 0.0006 | 0.9945 ± 0.0004 | 0.023 → 0.007 |
| efficientnet_b0 | CNN | group | lung_crop | none | 3 | 0.9515 ± 0.0016 | 0.9585 ± 0.0010 | 0.9934 ± 0.0002 | 0.025 → 0.008 |
| vit_b_16 | Transformer | group | none | none | 3 | 0.9502 ± 0.0030 | 0.9578 ± 0.0032 | 0.9936 ± 0.0002 | 0.028 → 0.012 |
| resnet50 | CNN | group | lung_crop | none | 3 | 0.9472 ± 0.0002 | 0.9537 ± 0.0002 | 0.9939 ± 0.0001 | 0.028 → 0.009 |
| swin_t | Transformer | group | lung_crop | none | 3 | 0.9487 ± 0.0016 | 0.9533 ± 0.0005 | 0.9937 ± 0.0004 | 0.026 → 0.009 |
| swin_t | Transformer | group | lungs_removed | none | 3 | 0.9329 ± 0.0025 | 0.9455 ± 0.0015 | 0.9902 ± 0.0003 | 0.026 → 0.009 |
| resnet50 | CNN | group | lungs_removed | none | 3 | 0.9312 ± 0.0042 | 0.9454 ± 0.0034 | 0.9903 ± 0.0005 | 0.027 → 0.011 |
| efficientnet_b0 | CNN | group | lungs_removed | none | 3 | 0.9304 ± 0.0038 | 0.9423 ± 0.0022 | 0.9903 ± 0.0007 | 0.024 → 0.011 |
| efficientnet_b0 | CNN | group | lungs_only | none | 3 | 0.9356 ± 0.0025 | 0.9398 ± 0.0022 | 0.9916 ± 0.0004 | 0.020 → 0.006 |
| resnet50 | CNN | group | lungs_only | none | 3 | 0.9326 ± 0.0048 | 0.9391 ± 0.0034 | 0.9910 ± 0.0002 | 0.029 → 0.009 |
| swin_t | Transformer | group | lungs_only | none | 3 | 0.9325 ± 0.0046 | 0.9360 ± 0.0052 | 0.9917 ± 0.0001 | 0.032 → 0.014 |
| efficientnet_b0 | CNN | group | none | no_pretrained | 2 | 0.9013 ± 0.0015 | 0.9083 ± 0.0030 | 0.9827 ± 0.0000 | 0.019 → 0.014 |

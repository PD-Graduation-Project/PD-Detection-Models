### Ideas
#### Datasets
- Augmentation with `albumentation` library
- Test using `CLAHE`/ histogram equalization for inference mode?
#### Models
[**Models and pre-trained weights**](https://docs.pytorch.org/vision/main/models.html)

- One for spiral one for waves, or, one model for both?❌ (dataset was mixed so one model for all)
- VGG16 / VGG19 (choose 19)
- ResNet-50
- InceptionV3
- DenseNet169 / DenseNet201 (choose 201)
- Maybe Vit-16b?
- Try deep supervision?❌ (only available in inception)
#### Validation rules
> Rules to follow for all experiments to be able to compare the models evenly.
- 5% of the dataset for sanity check.
- 20% of the dataset for validation and comparisons.
- 5 epochs for each model.
- Freezing all the model but the classification head, if more params are unfreezed it must be documented.

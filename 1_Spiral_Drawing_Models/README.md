> **Note:** Most notebook cell outputs have been cleared to keep the repository clean and easier to navigate. 
> 
> All code cells are preserved and fully executable. 
> 
> Key outputs, such as confusion matrices, are included in the README file, and training logs are available in the `results/` folder as text files.


# Phase ONE [**DONE**]
- All files created and working.
- Compared between all 5 models, accuracy and losses, when training them for:
    - 5 epochs
    - 10 epochs
- Finetuned the best model , `DenseNet50`, and trained it for extra 50 epochs. (found in 'experiments/finetuning/finetune_test1.iypnb')

## Best results from **PHASE ONE**
- Validation Accuracy: **87.7%**
- ![alt text](imgs/image.png)

## Conclusion
- Need to fix false negatives being more than false postives.
- The old loss function only had `BinaryCrossEntropy`, I couldn't work with `FocalLoss` or `TverskyLoss` from `monai.losses` as they were for segmentation, not classifcation.
- I have to measure *recall* for better evaluation of the model.

---

# Phase TWO [**DONE**]
- Updated the loss function to include **focal** and **tversky**, by manually getting their equations in code form.
- Updated the metric function to measure *recall*.
- Updated scheduler from `ReduceLROnPlateau` to `OneCycleLR` for more aggressive LR changes with fewer epochs.
- Increased inital LR from `1e-4` to `3e-4`.

## Best results from **PHASE TWO**
- Validation Accuracy: **89.37%**
- Validation Recall: **0.8407**
- Number of trained epochs: **15** (initial model comparisons) + **31** (fine-tuning with early stopping; planned 50).
- ![](imgs/PastedImage.png)

## Conclusion
- Changing the *scheduler* made the accuracy increase way faster.
- The new *loss function* needed many hyperparameters tuning, but now it is balanced.
- After only 10 epochs the model gave comparable outputs to the one trained for over 50 epochs. (found in 'experiments/finetuning/finetune_test2.iypnb')
- `DenseNet50` is still superior in accuracy, recall, and model size. 
- `VGG19_bn` is very promising, and has better recall, but the model is so much larger than `DenseNet50`. (may experiment with it more later)

---

# Phase THREE
- Updated the calssifier of the model, taking inspiration from [This paper](https://ieeexplore.ieee.org/document/10307641) (for more details see the `improved_classifier.md` file). This however increased the number of trainable parameters from `1,921` to `1,049,345`.
- Added gradient clipping to training loop.
- Updated max_lr from `3e-4` to `1e-3`.
- Added precision,and F1-score calculations.

## Best results from **PHASE THREE**
- Validation Accuracy: **97.02%**
- Validation Recall: **0.9787**
- Validation precision: **0.9609**
- Validation F1-Score: **0.9689**
- Number of trained epochs: **15** (initial model comparisons) + **43** (fine-tuning with early stopping; planned 50).
![alt text](imgs/image-1.png)

## Conclusion
- Massive improvement in accuracy and recall, along with a sharp drop in losses.
- Comaprison in accuracy and losses between phase 2 and phase 3:
    - ![](results/Phase%20THREE/finetuning/val_acc.png)
    - ![](results/Phase%20THREE/finetuning/val_loss.png)

## TO DO (for phase four maybe)
- Unfreeze some layers in the middle of the model or add more hidden layers.
- Compare `DenseNet50` with and without `CLAHE` in `val_dataloader`.
- Compare `DenseNet50` with the trained image size of `(224, 224)` and with `(512, 512)`.

# HIERVAR

Time series classification stands as a pivotal and intricate challenge across various domains, including finance, healthcare, and industrial systems. In contemporary research, there has been a notable upsurge in exploring feature extraction through random sampling. Unlike deep convolutional networks, these methods sidestep elaborate training procedures, yet they often necessitate generating a surplus of features to comprehensively encapsulate time series nuances. Consequently, some features may lack relevance to labels or exhibit multi-collinearity with others. In this paper, we propose a novel hierarchical feature selection method aided by ANOVA variance analysis to address this challenge. Through meticulous experimentation, we demonstrate that our method substantially reduces features by over 94\% while preserving accuracy-- a significant advancement in the field of time series analysis and feature selection.

## How to import

```python
# Import necessary libraries and modules from the 'hiervar' package
import hiervar.RASTER as RASTER
import hiervar.anova as anova
import hiervar.classifier as classifier
import hiervar.utils as utils
import numpy as np
import warnings
```

## apply it on MiniROCKET
```python
# Apply MiniROCKET transformation with 10,000 features
# 'shuffle_quant' is set to False to keep the quantization order fixed
x_train_trans_mini, x_test_trans_mini, parameter_raster = RASTER.MiniROCKET(
    x_train, y_train, x_test, y_test, n_features=10000, shuffle_quant=False
)

# Train a classifier on the transformed MiniROCKET data and evaluate its accuracy
accuracy_mini, _, clf = classifier.classic_classifier(
    x_train_trans_mini, y_train, x_test_trans_mini, y_test
)
print("Number of Features: ", x_train_trans_mini.shape[1], "Accuracy: ", accuracy_mini)

# Use HIERVAR to prune features from MiniROCKET-transformed data
result, erocket_index, selected_mean = anova.anova_erocket_pruner(
    x_train_trans_mini, y_train, threshold=None, divider=2, verbose=False
)


# Train the classifier again using only the pruned features
accuracy_erocket_modified, _, _ = classifier.classic_classifier(
    x_train_trans_mini[:, result], y_train, x_test_trans_mini[:, result], y_test
)
print("Number of Features (After HIERVAR): ", len(result), "Accuracy: ", accuracy_erocket_modified)
```

## Apply it on RASTER

```python 
# Apply the full RASTER transformation with the same number of features (10,000)
x_train_trans_raster, x_test_trans_raster, parameter_raster = RASTER.RASTER(
    x_train, y_train, x_test, y_test, n_features=10000, shuffle_quant=False
)

# Train a classifier on the RASTER-transformed data and evaluate its accuracy
accuracy_raster, _, clf = classifier.classic_classifier(
    x_train_trans_raster, y_train, x_test_trans_raster, y_test
)
print("Number of Features: ", x_train_trans_raster.shape[1], "Accuracy: ", accuracy_raster)

# Prune features again using HIERVAR on the RASTER-transformed data
result, erocket_index, selected_mean = anova.anova_erocket_pruner(
    x_train_trans_raster, y_train, threshold=None, divider=2, verbose=False
)

# Re-evaluate the classifier using only the pruned RASTER features
accuracy_erocket_modified, _, _ = classifier.classic_classifier(
    x_train_trans_raster[:, result], y_train, x_test_trans_raster[:, result], y_test
)
print("Number of Features (After HIERVAR): ", len(result), "Accuracy: ", accuracy_erocket_modified)
```

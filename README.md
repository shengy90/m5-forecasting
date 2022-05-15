# TODO:

### Calculate WRMSSE
- Test WRMSSE with calculated weights
    - We need to calculate weights using sales via the `calculate_weights` function
    - Merge it with dummy_preds and test assert

- Then import all of these into Kaggle and calculate naive forecas

### Calculate WRMSSE for each hierarchy
 - there are 12 hierarchies in total (as defined in `definitions.py`)
 - once we've calculate forecast for each id, we need to aggregate them for specified hierarchies K (1<=K<=12)
 - then calculate WRMSEE for for the K hierarchies as defined in the competition guide
 - we should output WRMSEE over all hierarchies and also for each hierarchy

### Cross Validation
- Then we can look at walk_forward/ sliding window cross validation


### Kaggle TO:DO:
- Perform Naive Forecaster in notebook and evaluate with WRMSSE for aggregation level 12

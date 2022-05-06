# TODO:

- Test WRMSSE with calculated weights
    - We need to calculate weights using sales via the `calculate_weights` function
    - Merge it with dummy_preds and test assert

- Then import all of these into Kaggle and calculate naive forecast

- Then we need to improve WRMSSE by calculating weights Weighted RMSSE for K hierarchies and average it by K

- And we should output also errors at each hierachy level 

- Then we can start with EDA and improve on naive forecast

- Then we can look at walk_forward/ sliding window cross validation


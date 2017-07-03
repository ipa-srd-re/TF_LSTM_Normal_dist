This tensorflow model will take some points as an input and outputs means and variances for a normal distribution corresponding to the most likely future positions.

If you want to fit your own distribution, please refer to the function tf_normal inside mdn_model. Also the function run_tests needs to be changed accordingly.

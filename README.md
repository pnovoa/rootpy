# Robust Optimization Over Time scenarios with PYthon (rootpy)

This project is devoted to provide examples about the performance of Differential Evolution algorithm (Storn and Price, 1995) on three different ROOT scenarios.

The scenarios are constructed from the presence of uncertainty in different components of the problem. They are as follows:

S1) Fitness function of future environments are known exactly, so the algorith can evaluate its solutions without any uncertainty.

S2) Fitness function of future environments are not know, but those from past environments are exactly known. So, past fitness function can be used to forecast the future.

S3) No fitness function from the past environments is exactly known. So, approximating (modeling) the past and the future environments is required in order to compute an estimation of Robustness.


Among the three, S3 is the most challenging.

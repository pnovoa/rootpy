# Robust Optimization Over Time scenarios with PYthon (rootpy)

This project is devoted to provide examples about the performance of Differential Evolution algorithm (Storn and Price, 1995) on three different scenarios of **robust optimization over time (ROOT)**.

A ROOT problem is defined as:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\max_{x\in\Omega}\mathcal{R}(x,t,T)" title="\Large \max_{x\in\Omega}\mathcal{R}(x,t,T)" />

where omega is the search space and a subset of the real-valued sapace in D dimension.

The scenarios are constructed from the presence (or not) of uncertainty in different components of the problem. They are as follows:

1. (S1) Fitness functions of future environments are known exactly, so the algorithm can evaluate its solutions without any uncertainty.

2. (S2) Fitness functions of future environments are not know, but those from past environments do. So, past fitness functions can be used to forecast the future.

3. (S3) No fitness functions from the past environments is exactly known. So, approximating (modeling) the past and the future environments is required in order to compute an estimation of robustness.

Among the three, S3 is the most challenging.

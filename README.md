# Gradient Decent Varients

This repository contains the comparison between the performance of various gradient decent algorithms. The data is randomly generated through numpy and is of size 200000 rows and 6 columns. The target is made such that it is having linear relation with the given columns. The bias and weights for each of the column is generated randomly and gradient decent is used to approximate these weights.

## Batch Gradient Decent
Batch Gradient Decent computes the cost function w.r.t parameter θ for the entire dataset.
> θ = θ − η·∇<sub>θ</sub>J(θ)

### Results:
 - Time taken: 46.77 sec
 - Number of epochs: 7193
Implementation of nGPT: [paper](https://arxiv.org/pdf/2410.01131)

## Introduction

The idea of the paper is to force vectors to be on a hypersphere instead of following the traditional Eucledian approach.

**Hypersphere**: An $n$-sphere, written $S^n$ is a set of (n + 1)-dimensional unit vectors. $$S_n = \{ x \in R^n : \lVert x \rVert = 1 \}$$ So, a hypersphere forces all the vectors to be on its surface. Since all the magnitudes are now 1, the only guiding measurement is the angle between vectors, which we can denote by similarity and is constrained between \[-1, 1]. 

Hypersphere, hence can be thought of as a different kind of regularization technique. But even more than just regularization, because the distance measurement between the vectors is only their angle (instead of their angle AND position), it is relatively easier to optimize things. (The paper claims that the speed increase is 4-20x which is mostly because of this optimization)

---

1. The learnable embedding matrix for both the input and output were unconstrained in the original transformer paper which can lead to difficult optimizations and not very representative similarity values. The authors propose to normalize the embedding vectors stored in the matrices after each step of training. 

2. Because the matrices are normalized, any vector multiplied by the matrix would result in a cosine similarity score which is constrained between \[-1, 1]. Because norm of all the vectors is 1, the optimization only requires to change the angle between vectors.

## Normalized Transformer
The normalized transformer performs a multistep optimization on a hyper plane where each step of Attention and MLP is controlled by an eigen learning rate.

For each token, the optimization path starts from its own embedding and goes to a point that best predicts the vector of the next token. (prediction).

To compute this, we first define `SLERP` which basically finds the shortest way to get from one point to another while being on the hyperplane
$$SLERP(a, b; \alpha) = \dfrac{sin((1 - \alpha) \theta)}{sin \theta} a + \dfrac{sin(\alpha \theta)}{sin(\theta)}b$$
where $\theta$ is the angle between a and b and $\alpha \in [0, 1]$ is a parameter. and $\alpha$ is the interpolation parameter. $\alpha = 0$  returns $a$ and $\alpha = 1$ returns b.

According to the experiments done by authors, this SLERP can be approximated to LERP (Linear interpolation) -- which effectively just joins the two points linearly (instead of following the curve)

$$LERP(a, b; \alpha) = (1 - \alpha) a + \alpha b$$

Okay, so to take a step back, LERP (and SLERP) allow to find a value that lies between two given values specified by some percentage i.e. $\alpha$.

Now, why do we care about any of this?

In nGPT, $a$ can be the hidden output $h$ and $b$ is the point suggested by the MLP or attention block (or what $a$ should've been for the proper ground truth value). Then, we would update $a$ using our LERP rule and the gradient $g = a - b$  such that
$$a \leftarrow a - \alpha B g$$
where B is the inverse of a Hessian matrix (a Hessian matrix is a square matrix that contains all the second-order partial derivatives of a scalar-valued function). Finally because of our constraint of everything to be a unit vector, we will normalize $a$ again. A good thing about this over something like RMSNorm or Layer Norm is that there are no element wise scaling products. NOTE: Even though the above method looks similar to how traditional update method works, it uses Riemannian Optimization which is an extension of Euclidean Space to Riemennean Manifolds.

$\alpha$ is also valled the eigen learning rate. The idea of an eigen learning rate is very similar to the traditional learning rate, however eigen learning rates are not constant and change based on the contour of the sphere. These learning rates are actually learnable parameters!

The authors also perform scaling in transformer block to aid the methods to utilize SiLU and other activation functions. 


## Implementation


## Results and Evaluation
**Condition Number**: of a function measures how much the output value of the function can change for a small change in the input argument.

**PG 19**


---
Other interesting reads: \
https://arxiv.org/abs/2407.09468 \
https://arxiv.org/pdf/2104.13478 \
https://geometricdeeplearning.com 
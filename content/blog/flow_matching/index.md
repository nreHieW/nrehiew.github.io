+++
title = 'Flow Matching in 3 Minutes'
date = 2024-12-06
[params]
subtitle = "Busy person's intro to Flow Matching"
+++
# Flow Matching in 3 Minutes
#### *Busy person's intro to Flow Matching*

In this post, I will try to build an intuitive understanding to Flow Matching, a framework used to train many state-of-the-art generative image models. 

We start with 2 probability distributions $p_{\text{source}}$ and $p_{target}$ and our goal is to transform a point from $p_{\text{source}}$ to a point that could have been reasonably sampled from $p_\text{target}$. 

## 2D Example
Suppose $p_{\text{source}}$ and  $p_{target}$ are isotropic Gaussian distributions centred at $(0, -5)$ and $(5, 0)$ respectively. We sample one point from each distribution which happen to result in the mean points $(0, -5)$ (source) and $(5, 0)$ (target).

### How do we move from the source to the target?

The simplest approach is a straight-line trajectory which minimizes the total distance we need to travel. We can take the vector difference between source and target $(5,0) - (0,-5) = (5,5)$ which represents the total "direction" and "magnitude" of movement required.

If we want to move in a single step, we move directly by $(5,5)$. But suppose we want to take five steps instead. We can decompose our direction of movement $(5,5) = 5 * (1,1)$, where we travel in the direction of $(1,1)$ five times. For five steps, the trajectory looks like:
$$
(0, -5) \rightarrow (1, -4) \rightarrow (2, -3) \rightarrow (3, -2) \rightarrow (4, -1) \rightarrow (5, 0)
$$
Notice that at each step, the direction of movement $(1,1)$ remains consistent.

### Understanding Intermediate Steps

Now, consider an intermediate step, such as step 3 of 5. First, we must determine our current location. Since the motion is along a straight line and the source and target points are known, we can calculate the position by interpolating between them. To simplify, we normalize the interpolation to the range $[0, 1]$
$$x_t = x_{3/5} = (1-\frac{3}{5}) (0, -5) + (\frac{3}{5}) (5, 0) = (0, -2) + (3, 0) = (3, -2)$$ 

Note that $(3,-2)$ is not a point directly sampled from either $p_{\text{source}}$ or $p_{target}$ but rather lies somewhere between the two distributions. The timestep $t=3/5$ helps us identify where we are in the transition. Critically, regardless of the timestep, the direction of movement remains consistent at $(1,1)$

This is basically all there is to flow matching! I've provided the code below which matches the intuition we built above.

```python
# a single training step 
t = torch.rand(1) # sample a single (normalized) timestep between [0, 1)
intermediate_t = (1-t) * source + (t * target) # current position at timestep t 
direction = target - source # this is the direction we always have to move 
# we give the model the timestep information so it knows where between source and target we are at 
prediction = model(intermediate_t, t) 
loss = ((direction - pred) **2).mean() # standard regression 
loss.backward() 
```
*Formally, what we called "direction" $(1,1)$ is the rate of change of $x$ with respect to time $t$: $\frac{dx_t}{dt}$. From the interpretation $x_t= (1-t)x_0 + t x_1$, differentiating gives 
$$\frac{dx_t}{dt} = -x_0 + x_1 = x_1 - x_0$$ Thus, $\frac{dx_t}{dt}$​​ is equivalent to the direction of movement from $x_0$​ (source) to $x_1$​ (target)*

### Sampling
Typically $p_{source}$ is a distribution we can sample easily from (eg. Gaussian) while we cannot do the same for the more complex $p_{target}$. We usually only have data samples from $p_{target}$ which we use for training. To sample from $p_{target}$, we need to start from a source point sampled from $p_{source}$ and iteratively move in the predicted direction. For a trajectory with `NUM_STEPS`, we scale the prediction each iteration:

```python
source = torch.randn(1) # or any other source distribution
for t in range(NUM_STEPS):
	prediction = model(source, t) # predict the direction to move
	source = source + (1 / NUM_STEPS) * prediction # scale and move 
source # this should be the target point 
```

### High-Dimensional Case
This intuition directly extends to higher dimensions. For example, in image generation, the source could be a high-dimensional Gaussian distribution ($H * W * 3$) and the target an image. The flow-matching process transforms samples from the source distribution (for eg. random noise) into samples resembling real images.

### References 
Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. arXiv. [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)

Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv. [https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
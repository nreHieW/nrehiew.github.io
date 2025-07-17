+++
title = 'Flow Matching in 5 Minutes'
date = 2025-07-15
[params]
subtitle = "Busy person's intro to Flow Matching"
math = true
+++
# Flow Matching in 5 Minutes
###  *Busy person's intro to Flow Matching*

In this post, I will try to build an intuitive understanding to the Flow Matching, a framework used to train many state-of-the-art generative image models. 

In generative modelling, we start with 2 probability distributions: (1) an easily sampled distribution $p_{\text{source}}$ (e.g. a Gaussian distribution) and (2) a target distribution $p_{target}$ containing data points (e.g. images). Our goal is to transform a point sampled from $p_{\text{source}}$ to a point that could have been reasonably sampled from $p_\text{target}$. 


## 2D Example
We sample one point from each distribution, resulting in the points $(1, 1)$ (source) and $(6, 6)$ (target) respectively and we want to find a way to map the source point to the target point. One way to model this transformation is as a trajectory made up of multiple steps across time where starting from the source point, we move in the direction of the target point at every step.

**How do we move from the source to the target?**

There are infinitely many paths we can take but the simplest approach is simply move in a straight line. We can use the vector difference between source and target $(6,6) - (1,1) = (5,5)$ which represents the total "direction" and "magnitude" of movement required.

If we want to move from $(1,1)$ to $(6,6)$ in a single step, we move directly by $(5,5)$. But suppose we want to take multiple iterative steps instead. If we want to take 5 total steps, we can decompose our direction of movement $(5,5) = 5 * (1,1)$, where we travel in the direction of $(1,1)$ five times. For five steps, the trajectory looks like this:
$$
(1, 1) \rightarrow (2, 2) \rightarrow (3, 3) \rightarrow (4, 4) \rightarrow (5, 5) \rightarrow (6, 6)
$$
Notice that since this is a straight-line, **at each step, the direction of movement $(1,1)$ remains the same**.

**Understanding Intermediate Steps**

Now, consider an intermediate step, say step 3 of 5. We must first determine our current location. Since the motion is along a straight line and the source and target points are known, we can calculate the position by interpolating between source and target.

For simplicity, we normalize the interpolation to the range $[0, 1]$. With $t=3/5$, our current position $x_t = (1-t)x_0 + t x_1 = (1-3/5) * (1,1) + (3/5) * (6,6) = (4,4)$.

<div style="display: flex; justify-content: center; align-items: center;">
  <div style="text-align: center;">
    <video width="600" autoplay muted loop>
      <source src="assets/FlowMatching.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div style="margin-top: 10px; font-size: 14px">Flow Matching</div>
  </div>
</div>
<br>

Notice that $(4,4)$ is unlikely (low probability) to have been directly sampled from either $p_{\text{source}}$ or $p_{\text{target}}$ but rather lies somewhere 'between the two distributions'. The timestep $t=3/5$ helps us identify where we are in this straight line trajectory. 

This is the key idea and what we are trying to learn. At each time step, we provide the model with the current position and the timestep, and we want it to predict the direction of movement that will get us closer to some point in the target distribution. To train the model, we can optimize against the ground truth (straight line) direction of movement which is just the vector difference between the source and target points.

## Vector Fields
Recall that our goal is to learn how to go from the source distribution $p_\text{source}$ to the target distribution $p_\text{target}$. Specifically for every point in the space, we want to learn a direction and magnitude of movement to follow at every timestep. That is, given the current timestep and position, the model should predict the direction and magnitude of where to move next. Since we want to learn this direction for every single point in space, this can be modelled as a vector field for each timestep.

However, during training, we often don't have $p_\text{target}$ available and instead only have data points sampled from $p_\text{target}$. We also don't have access to the ground truth vector field. The solution is to work with individual pairs of points (1 from source and 1 from target) - just like our example of moving from $(1,1)$ to $(6,6)$. This raises two questions. First, how does training on these individual trajectories help us learn a transformation that works for the entire distribution? Second, would't trajectories that intersect create conflicting directions of movements at the intersection points?

It turns out that even though we train on individual trajectories, the model learns a consistent vector field that can transform any point from the source distribution to the target distribution. Instead of memorizing specific paths, it learns to approximate an "average" field that works for the entire distribution rather than memorizing specific paths between individual points.<sup>[1](#footnote1)</sup>

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

>**Some derivation**
<br>
> Formally, what we called "direction" $(1,1)$ is the rate of change of $x$ with respect to time $t$, $\frac{dx}{dt}$
> From the interpretation that any intermediate point $x_t$ is a linear interpolation between $x_0$ (source) and $x_1$ (target) (i.e. $x_t= (1-t)x_0 + t x_1$), differentiating gives 
> $$\frac{dx_t}{dt} = -x_0 + x_1 = x_1 - x_0$$
> Thus, predicting $\frac{dx_t}{dt}$ is equivalent to predicting the direction of movement from $x_0$ to $x_1$ or in our 2D example, the simple vector difference $(6,6) - (1,1) = (5,5)$.
   
## Sampling
To sample from $p_{\text{target}}$, we start from a source point and iteratively move in the predicted direction. For a trajectory with `NUM_STEPS`, if we assume that the entire trajectory consists of uniform steps with an interval of `1/NUM_STEPS`:

```python
source = torch.randn(1) # or any other source distribution
current_position = source
for t in range(NUM_STEPS):
	prediction = model(current_position, t) # predict the direction to move
	current_position = current_position + (1 / NUM_STEPS) * prediction # move based on the step size 
current_position # the target point 
```

## High-Dimensional Case
This intuition directly extends to higher dimensions. For example, in image generation, the source could be a high-dimensional Gaussian distribution ($H * W * 3$) that we can easily sample from and the target an image from the dataset distribution. Then, the flow-matching process transforms samples from the source distribution into samples resembling the target.


## References 
Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). _Flow matching for generative modeling_. arXiv. [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)

Liu, X., Gong, C., & Liu, Q. (2022). _Flow straight and fast: Learning to generate and transfer data with rectified flow_. arXiv. [https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)

Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R. T. Q., Lopez-Paz, D., Ben-Hamu, H., & Gat, I. (2024). _Flow matching guide and code_. arXiv. [https://arxiv.org/abs/2412.06264](https://arxiv.org/abs/2412.06264)

The references above are some of the more seminal/popular works on flow matching. 


The point of this post was to show that the core idea really is not that difficult or scary and to see if I could share some of my intuitions within a concise/constrained post. If you think this was interesting there are a ton of other blogs and videos that have already covered this topic in more detail.

A non-exhaustive list:
1. ["An Introduction To Flow Matching"](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) by Tor Fjelde, Emile Mathieu, Vincent Dutordoir

2. ["Diffusion Meets Flow Matching: Two Sides of the Same Coin"](https://diffusionflow.github.io/) by Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, Tim Salimans (Google DeepMind)

3. ["How I Understand Flow Matching"](https://www.youtube.com/watch?v=DDq_pIfHqLs) by Jia Bin Huang

4. ["A Visual Dive into Conditional Flow Matching"](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/) by Anne Gagneux (INRIA, ENS de Lyon), Ségolène Martin (Technische Universität Berlin), Rémi Emonet (Inria, Université Jean Monnet), Quentin Bertrand (Inria, Université Jean Monnet), Mathurin Massias (Inria, ENS de Lyon)

5. ["Flow With What You Know"](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-flow-with-what-you-know-38/blog/flow-with-what-you-know/) by Scott H. Hawley, ICLR Blogposts, 2025

---

<a id="footnote1"></a>**[1]** If you want the full derivation of why the aggregate/expectation of the conditional probability paths form the full unconditional velocity field, refer to sections 4.2 and 4.4 of [Lipman et al, 2024](https://arxiv.org/abs/2412.06264).

_Huge thanks to [Sander Dieleman](https://x.com/sedielem) for the feedback and help with early drafts of this post!_
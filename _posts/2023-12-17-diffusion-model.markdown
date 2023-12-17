---
layout: post
title: "Diffusion Model"
subtitle: 'My notes for the Diffusion Model'
author: "Jiaqing Zhang"
header-style: text
tags:
  - diffusion
  - deep learning
  - generative model
---

# Diffusion Model

#Model 

## Introduction

**likelihood-based models**, which directly learn the distribution’s probability density (or mass) function via (approximate) maximum likelihood.
**implicit generative models**, where the probability distribution is implicitly represented by a model of its sampling process.

GLIDE [<sup>1</sup>](#reference-1), DALLE2[<sup>2</sup>](#reference-2), Imagen[<sup>3</sup>](#reference-3) are really popular in text-to-image task

## Method
### Generative models
[GAN](https://lilianweng.github.io/posts/2017-08-20-gan/), [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/), [Flow-based](https://lilianweng.github.io/posts/2018-10-13-flow-models/) are good but: 
- GAN: <font color=red>unstable training</font> and <font color=red>less diversity</font> in generation due to their adversarial training nature. mode collapse.
- VAE relies on a surrogate loss to approximate maximum likelihood training
- Flow models have to use specialized architectures to construct reversible transform or to ensure a tractable normalizing constant for likelihood computation

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).

![[Diffusion learning fig 1.png]]

### Diffusion model  [(Ho et al, NeurIPS 2020)](https://arxiv.org/pdf/2006.11239.pdf)
**Forward diffusion process** (Adding noise to the data)
Given a data point sampled from a real data distribution $x_0∼q(x)$, let us define a _forward diffusion process_ in which we add small amount of Gaussian noise (噪声的标准差是固定值$\beta_t$，均值是以$\beta_t$和当前时刻$t$的数据$x_t$决定的，所以这个噪声不含参) to the sample in $T$ steps using first order Markov property, producing a sequence of noisy samples $x_1,…,x_T$. The step sizes are controlled by a variance schedule ${β_t∈(0,1)}_{t=1}^T$.
$$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$

$$q(x_{1:T}|x_0)=\prod\limits_{t=1}^Tq(x_t|x_{t-1})$$
The data sample $x_0$ gradually loses its distinguishable features as the step $t$ becomes larger. Eventually when $T→∞$, $x_T$ is equivalent to an isotropic Gaussian distribution.

![[Diffusion learning fig 2.png]]

A nice property of the above process is that we can sample $x_t$ at any arbitrary time step $t$ in a closed form using [reparameterization trick](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick) (不需要迭代，可以通过$x_0$和$\beta_t$算出来). Let $α_t=1−β_t$ and $\overline{a}_t=\prod\limits_{i=1}^Tα_i$:

![[Screen Shot 2022-10-05 at 21.45.49.png]]

![[Diffusion learning fig 4.png]]

$(*)$ Recall that when we merge two Gaussians with different variance, $\mathcal{N}(0,\sigma^2_1I)$ and $\mathcal{N}(0,\sigma^2_2I)$, the new distribution is $\mathcal{N}(0,(\sigma^2_1+ \sigma^2_2)I)$. (合并后的标准差为$\sqrt{(1-\alpha_t)+\alpha_t(1-\alpha_{t-1})}=\sqrt{1-\alpha_t\alpha_{t-1}}$)

Usually, we can afford a larger update step when the sample gets noisier, so $β_1<β_2<⋯<β_T$ and therefore $\overline{α}_1>⋯>\overline{α}_T$.

总结：1）不含参，2）任意时刻被推断出来，即使不迭代，3）$q(x_0)$ is the true data distribution，4）$p(x_0)$ is the model，5）q是扩散过程，p是逆扩散过程

Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, _stochastic gradient Langevin dynamics_ ([Welling & Teh 2011](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.226.363)) can produce samples from a probability density $p(x)$ using only the gradients $∇_xlog⁡p(x)$ in a Markov chain of updates:
$$x_t=x_{t−1}+δ/2∇_xlog⁡q(x_{t−1})+\sqrt{δ}ϵ_t$$

where δ is the step size, $ϵ_t∼\mathcal{N}(0,I)$. When $T→∞$, $ϵ→0$, $x_T$ equals to the true probability density $p(x)$.

Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima.

**Reverse diffusion process**
If we can reverse the above process and sample from $q(x_{t−1}|x_{t})$, we will be able to recreate the true sample from a Gaussian noise input, $x_T~\mathcal{N}(0,I)$.
Because it is hard, learn a model $p_\theta$ to approximate conditional probabilities  $q(x_{t-1}|x_t)$.
$$p_\theta(x_{0:T})=p(x_T)\prod\limits_{t=1}^Tp_{\theta}(x_{t-1}|x_t)$$
$$p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$
When conditioned on $x_0$ (后验的扩散条件概率):
$$q(x_{t-1}|x_t,x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}(x_t,x_0),\tilde{\beta_t}I)$$
Using Bayes' rule:
![[Screen Shot 2022-10-06 at 00.03.52.png]]
- 第一行 to 第二行：因为是Markov Chain，所以$x_0$无关，且$q(x_{t}|x_{t-1})$是一个均值为$\sqrt{1-\beta_t}x_{t-1}$，方差为$\beta_t$的分布
- 第二行 to 第三行：化简
- 第三行 to 第四行：$ax^2+bx=a(x+b/2a)^2+C$

where $C(x_t,x_0)$ is some function not involving $x_t−1$ and details are omitted. Following the standard Gaussian density function, the mean and variance can be parameterized as follows (recall that $α_t=1−β_t$ and $\bar{α}_t=∏_{i=1}^Tα_i$):
![[Screen Shot 2022-10-06 at 00.26.07.png]]
where $\tilde{\beta}_t$ 就是个常数

We can represent $x_0=1/\sqrt{\bar{\alpha}_t}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)$

![[Screen Shot 2022-10-06 at 11.45.09.png]]

Plan A
Deriving the ELBO using Jensen’s inequality
Say we want to minimize the cross entropy as the learning objective,

![[Screenshot 2022-10-27 at 12.33.51.png]]


Plan B
Similar to [[VAE]] and thus using the variational lower bound to optimize the negative log-likelihood. 在负对数似然函数的基础上加一个KL散度($>=0$)，就构成higher bound了

![[Screen Shot 2022-10-06 at 11.46.34.png]]

获得了交叉熵的上界之后，对其进行化简：
- 第四行 to 第五行：Markov property of the forward process, and the Bayes’ rule. Due to Markov Chain, adding $x_0$ doesn't matter
![[Screen Shot 2022-10-26 at 18.07.57.png]]

![[Screen Shot 2022-10-06 at 12.44.06.png]] 
- 第五行 to 第六行：第三项，$t=2$ to $T$ 求和，分母分子很多项可以约掉，只剩第六行的
- 第六行 to 第七行：分子分母互相约

![[Screen Shot 2022-10-06 at 13.12.41.png]]
$L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $x_T$ is a Gaussian noise. 方差又是常数，所以可训练参数只存在于均值中  

1. **Parameterization of $L_t$ for Training Loss**

Recall that we need to learn a neural network to approximate the conditioned probability distributions in the reverse diffusion process

Parameterize $L_t$ to minimize the difference from $\tilde{\mu}$ (两个高斯分布的KL散度$KL(p,q)=log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}$): 
![[Screenshot 2022-10-27 at 12.55.48.png]]
参数重整化![[Screenshot 2022-10-27 at 13.30.25.png]]

优化 1
Empirically, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) found that training the diffusion model works better with a simplified objective that ignores the <font color=orange>weighting</font> term, so we have objective function (优化这个就是优化负对数似然):
$$L_t^{simple}=\mathbb{E}_{t\sim[1,T],x_0,\epsilon}[||\epsilon_t-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t,t)||^2]$$
![[Screenshot 2022-10-27 at 13.01.03.png]]

优化 2
Connection with noise-conditioned score networks (NCSN)

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed a score-based generative modeling method where samples are produced via [Langevin dynamics](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) using gradients of the data distribution estimated with score matching. The score of each sample $x$’s density probability is defined as its gradient $∇xlog⁡q(x)$. A score network $s_θ:\mathbb{R}^D→\mathbb{R}^D$ is trained to estimate it, $s_θ(x)≈∇xlog⁡q(x)$.

2. **Parameterization of $\beta_t$**

3. **Parameterization of reverse process variance $\sum_\theta$**

## Discussion
- **Pros**: Tractability and flexibility are two conflicting objectives in generative modeling. Tractable models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. Flexible models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. Diffusion models are both analytically tractable and flexible
- **Cons**: Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite expensive in terms of time and compute. New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.


## Development 
![[Diffusion learning fig 5.png]]

## Generic framework
![[Diffusion learning fig 6.png]]

### _Denoising diffusion probabilistic model_ (**DDPM**; )

### _Noise conditioned score networks_ (**NCSNs**; )

### _Stochastic differential equations_ (**SDEs**;)

## Categorization
1. Unconditional Image Generation
2. Conditional Image Generation
3. Text-to-Image Synthesis
4. Image Super-Resolution
5. Image Editing
6. Image Inpainting
7. Image-to-Image Translation
8. Image Segmentation
9. Multi-Task Approaches
10. Medical Image Generation and Translation
11. Anomaly Detection in Medical Images
12. Video Generation



![[Diffusion learning fig 3.png]]



## Reference
<div id="reference-1"></div>
1. [Glide: Towards photorealistic image generation and editing with text-guided diffusion models.](https://arxiv.org/pdf/2112.10741.pdf)
<div id="reference-2"></div>
2. [DALLE2](https://openai.com/dall-e-2/)
<div id="reference-3"></div>
3. [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.](https://arxiv.org/pdf/2205.11487.pdf)


# Review
## In healthcare
[Adapting pretrained vision-language foundational models to medical imaging domains](https://arxiv.org/abs/2210.04133)

[**Diffusion**-based Data Augmentation for Skin Disease Classification: Impact Across Original Medical Datasets to Fully Synthetic Images](https://arxiv.org/abs/2301.04802)

[Bi-parametric prostate MR image synthesis using pathology and sequence-conditioned **stable diffusion**](https://arxiv.org/abs/2303.02094)

[Brain **imaging** generation with latent **diffusion** models](https://link.springer.com/chapter/10.1007/978-3-031-18576-2_12)

[**Diffusion** Probabilistic Models beat GANs on **Medical Images**](https://arxiv.org/abs/2212.07501)

[MedSegDiff: **Medical Image** Segmentation with **Diffusion** Probabilistic Model](https://arxiv.org/abs/2211.00611)

[MedSegDiff-V2: **Diffusion** based **Medical Image** Segmentation with Transformer](https://arxiv.org/abs/2301.11798)

[Spot the fake lungs: Generating synthetic **medical images** using neural **diffusion** models](https://link.springer.com/chapter/10.1007/978-3-031-26438-2_3)

[A New Chapter for **Medical Image** Generation: The **Stable Diffusion** Method](https://ieeexplore.ieee.org/abstract/document/10049010/?casa_token=zPE2vkBYMTQAAAAA:0eZOzxWa5IT0gY48xEZMBzul3FTya2C2fq8QPQprm1cg1z2Z8qSXMvixWrfFW1RMc8iOoTg5)
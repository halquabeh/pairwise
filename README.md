# Pairwise Learning via Stagewise Training in Proximal Setting
The pairwise objective paradigms are an important and essential aspect of machine learning.
Examples of machine learning approaches that use pairwise objective functions include differential network in face recognition, metric learning, bipartite learning, multiple kernel learning, and maximizing of area under the curve (AUC). Compared to pointwise learning, pairwise learning's sample size grows quadratically with the number of samples and thus its complexity.
Researchers mostly address this challenge by utilizing an online learning system. Recent research has, however, offered adaptive sample size training for smooth loss functions as a better strategy in terms of convergence and complexity, but without a comprehensive theoretical study. 
In a distinct line of research, importance sampling has sparked a considerable amount of interest in finite pointwise-sum minimization. This is because of the stochastic gradient variance, which causes the convergence to be slowed considerably. 
In this paper, we combine adaptive sample size and importance sampling techniques for pairwise learning, with convergence guarantees for nonsmooth convex pairwise loss functions. In particular, the model is trained stochastically using an expanded training set for a predefined number of iterations derived from the stability bounds.
In addition, we demonstrate that sampling opposite instances at each iteration reduces the variance of the gradient, hence accelerating convergence. 
Experiments on a broad variety of datasets in AUC maximization confirm the theoretical results. 

# How to Use the Project
The experiements are solely peroformed on AUC maximization on different datasets, using ouralgorithm and one of the state-of-art algorithm SPAM (Natole, M & Ying, Y., & Lyu,S. 2018 PMLR.) with proximal net as non-smooth regularization function. Given a space $\mathcal{X}\times \mathcal{Y}$ with unknown distribution $\mathcal{P}$ and $\mathcal{Y} = \{+1,-1\}$. A function that takes a sample,e.g. $x\in \mathcal{X}$ drawn independently according to $\mathcal{P}$ and predicts the classes $f:\mathcal{X} \rightarrow \mathcal{Y} $ have AUC score given by:

$AUC(f) := Pr(f(x)>f(x')|y=1,y'=-1)  = E[I_{f(x)>f(x')}|y=1,y'=-1]$

where the expectation is w.r.t. the samples. 
However given the fact that the formulation above is neither convex nor differentiable, researchers have approximated the AUC by using surrogate convex functions, such as the square function and hinge loss. In addition to the fact that data are often limited and the distribution is unidentifiable, it is impossible to calculate the expectation directly; as a result, we present the empirical AUC score in following with linear model and regularized squared loss function.

$AUC(w) = \frac{1}{2n^+ n^-} \sum_{i\in[n^+],j\in[n^-]} (1 - [w^T(x_i^+  - x_j^-)])^2 + \lambda \Omega(w)$

where $x_i^+$, $x_i^-$ denote the positive and negative examples respectively, $w\in \mathbb{R}^d$ is d-dimensional linear model weight and $\Omega(w)$ is non-smooth but convex regularization. 
The elastic net  i.e.

$\Omega(w) = \lambda \|w\|^2 + \lambda_1 \|w\|_1$

is considered to have fair compassion with AUC maximization algorithms in literature. However, $l_1$ (lasso) or mixture of both $l_1$ and $l_2$ (group lasso) can be applied. 

# Running The Experiemnts

1- Download the data of interest from LIBSVM 
{https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/}


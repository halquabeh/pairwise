# Pairwise Learning via Stagewise Training in Proximal Setting
The pairwise objective paradigms are an important and essential aspect of machine learning.
Examples of machine learning approaches that use pairwise objective functions include differential network in face recognition, metric learning, bipartite learning, multiple kernel learning, and maximizing of area under the curve (AUC). Compared to pointwise learning, pairwise learning's sample size grows quadratically with the number of samples and thus its complexity.
Researchers mostly address this challenge by utilizing an online learning system. Recent research has, however, offered adaptive sample size training for smooth loss functions as a better strategy in terms of convergence and complexity, but without a comprehensive theoretical study. 
In a distinct line of research, importance sampling has sparked a considerable amount of interest in finite pointwise-sum minimization. This is because of the stochastic gradient variance, which causes the convergence to be slowed considerably. 
In this paper, we combine adaptive sample size and importance sampling techniques for pairwise learning, with convergence guarantees for nonsmooth convex pairwise loss functions. In particular, the model is trained stochastically using an expanded training set for a predefined number of iterations derived from the stability bounds.
In addition, we demonstrate that sampling opposite instances at each iteration reduces the variance of the gradient, hence accelerating convergence. 
Experiments on a broad variety of datasets in AUC maximization confirm the theoretical results. 

# How to Use the Project
The experiements are solely peroformed on AUC maximization on different datasets, using ouralgorithm and one of the state-of-art algorithm SPAM (Natole, M & Ying, Y., & Lyu,S. 2018 PMLR.) with proximal net as non-smooth regularization function.

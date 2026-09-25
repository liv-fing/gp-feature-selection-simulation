# An Investigation of Bayesian Lasso Priors for Linear Mean Gaussian Process Regression

Livia Fingerson

Independent Study in Industrial Engineering

Advised by Professor Moses Y. -H. Chan

June 12, 2026

## Description 
This study investigates whether Bayesian Lasso priors, alone or in combination with orthogonal Gaussian process methods,
can recover interpretability from a linear mean function within Gaussian process regression models. 

## Repository Structure

## Methods


## Data
Models were tested on a dataset from a diabetes study from Efron et al. (2004). This dataset has 442
observations, corresponding to individual patients. Each patient had 10 baseline measures recorded: age,
sex, body mass index, blood pressure, and six blood serum measurements. The response variable corresponds
to a measure of diabetes progression one year after the initial measurements were observed. Features are
centered around the mean and scaled by the standard deviation multiplied by the square root of n = 442
(Efron et al. 2004). Pre-standardized data was loaded directly via a scikit-learn Python package (Pedregosa
et al. 2015). Following the structure in Park and Casella (2008), the response variable was centered about
the mean so that the intercept could be marginalized, which simplifies computation. Models were trained
on a random split, with 80% of the data used for training and 20% reserved to test each model’s ability to
generalize to unseen data.

(synthetic data)

## References
Abril-Pla, Oriol et al. (Sept. 2023). “PyMC: A Modern, Comprehensive Probabilistic Programming Frame-
work in Python”. In: PeerJ Computer Science 9, e1516. doi: 10.7717/peerj-cs.1516.

Ashton, Gregory et al. (Apr. 2019). “Bilby: A User-Friendly Bayesian Inference Library for Gravitational-
Wave Astronomy”. In: The Astrophysical Journal Supplement Series 241.2, p. 27. doi: 10.3847/1538-
4365/ab06fc. url: https://arxiv.org/abs/1811.02042.

Efron, Bradley et al. (Apr. 2004). “Least Angle Regression”. In: The Annals of Statistics 32.2, pp. 407–499.
doi: 10.1214/009053604000000067. url: https://projecteuclid.org/euclid.aos/1083178935.

Gelman, Andrew et al. (2014). Bayesian Data Analysis. 3rd ed. Boca Raton: CRC Press. isbn: 9781439898208.
Park, Trevor and George Casella (June 2008). “The Bayesian Lasso”. In: Journal of the American Statistical
Association 103.482, pp. 681–686. doi: 10.1198/016214508000000337.

Pedregosa, F. et al. (June 2015). “Scikit-learn”. In: GetMobile: Mobile Computing and Communications 19.1,
pp. 29–33. doi: 10.1145/2786984.2786995.

Plumlee, Matthew and Joseph V. Roshan (2016). Orthogonal Gaussian Process Models. arXiv: 1611.00203.
url: https://arxiv.org/abs/1611.00203.

Rasmussen, Carl Edward and Christopher K. I. Williams (2006). Gaussian Processes for Machine Learning.
Cambridge, MA: MIT Press. isbn: 9780262182539.

Vehtari, Aki et al. (July 2020). “Rank-Normalization, Folding, and Localization: An Improved ˆR for Assessing
Convergence of MCMC”. In: Bayesian Analysis. doi: 10.1214/20-BA1221.
‌

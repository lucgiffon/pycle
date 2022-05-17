from pycle.sketching.sigma_estimation import estimate_Sigma
from pycle.utils.datasets import generatedataset_GMM


def test_estimate_Sigma():
    d = 2  # Dimension
    K = 4  # Number of Gaussians
    n = 20000  # Number of samples we want to generate

    (X, GT_GMM) = generatedataset_GMM(d, K, n, normalize='l_inf-unit-ball', output_required='GMM', balanced=0.1,
                                      covariance_variability_inter=1.5)

    m0 = 200    # use a pre-sketch of size 100
    n0 = n//50  # observe 2% of the dataset to estimata Sigma

    Sigma = estimate_Sigma(X, m0, c=10, n0=n0, verbose=1)
    print("Estimated sigma2_bar: ", Sigma[0][0])

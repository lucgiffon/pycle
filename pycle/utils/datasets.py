import numpy as np
import scipy.stats


############################
# DATASET GENERATION TOOLS #
############################

def generatedataset_GMM(d, K, n, output_required='dataset', balanced=True, normalize=None, grid_aligned=True, **generation_params):
    """
    Generate a synthetic dataset according to a Gaussian Mixture Model distribution.

    Parameters
    ----------
    d: int, the dataset dimension
    K: int, the number of Gaussian modes
    n: int, the number of elements in the dataset (cardinality)
    output_required: string (default='dataset'), specifies the required outputs (see below). Available options:
       - 'dataset': returns X, the dataset;
       - 'GMM': returns (X,GMM), where GMM = (weigths,means,covariances) is a tuple describing the generating mixture;
       - 'labels': returns (X,y), the dataset and the associated labels (e.g., for classification)
       - 'all': returns (X,y,GMM)
    balanced:  - bool (default=True), if True the Gaussians have the same weights, or
               - real (must be > 0.), stength of weight imbalance (~0 is very imbalanced, > K is fairly balanced)
    normalize: string (default=None), if not None describes how to normalize the dataset. Available options:
            - 'l_2-unit-ball': the dataset is scaled in the l_2 unit ball (i.e., all l_2 norms are <= 1)
            - 'l_inf-unit-ball': the dataset is projected in the l_inf unit ball (i.e., all entries are <= 1)
    grid_aligned: bool (default = True), if True the covariances of the GMM modes are diagonal



    Returns
    -------
    out: array-like or tuple, a combination of the following items (see desciption of output_required):
        - X: (n,d)-numpy array containing the samples; only output by default
        - weigths:     (K,)-numpy array containing the weigthing factors of the Gaussians
        - means:       (K,d)-numpy array containing the means of the Gaussians
        - covariances: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
        - y: (n,)-numpy array containing the labels (from 0 to K, one per mode) associated with the items in X


    Other Parameters
    ----------------
    separation_scale: scale driving the average separation of the Gaussians (larger for well-separated modes)
    separation_min: minimal distance separation to impose between the centers
    covariance_variability_inter: diversity of Gaussian covariances between the clusters (if = 1 all clusters have the same covariance matrix scale, if > 1 the clusters have covariances of different scales)
    covariance_variability_intra: diversity of the covariance matrix inside each cluster (if = 1 all dimensions have the same variance (isotropic Gaussian), if > 1 the covariance matrix has different entries for each dimension)
    all_covariance_scaling: a re-scaling factor that re-scales the covariance of all Gaussians (redundant with separation_scale)

    """
    ## STEP 0: Parse input generation parameters
    # Default generation parameters
    _gen_params = {
        'separation_scale': (10/np.sqrt(d)), # Separation of the Gaussians
        'separation_min': 0, # Before norm
        'covariance_variability_inter': 1., # between clusters
        'covariance_variability_intra': 1., # inside one mode
        'all_covariance_scaling': 0.15}
    # Check the inputs, if it's a valid parameter overwrite it in the internal parameters dict "_gen_params"
    for param_name in generation_params:
        if param_name in _gen_params.keys():
            _gen_params[param_name] = generation_params[param_name]
        else:
            raise ValueError('Unrecognized parameter: {}'.format(param_name))
    if _gen_params['separation_min'] > 2 * _gen_params['separation_scale']:
        print("WARNING: minimum separation too close to typical separation scale, finding separated clusters might be hard")

    ## STEP 1: generate the weights of the Gaussian modes
    # Convert input to a "randomness strength"
    if isinstance(balanced,bool):
        weight_perturbation_strength = 0. if balanced else 3.
    else:
        weight_perturbation_strength = 1./balanced
    # Generate random weigths, normalize
    weights = np.ones(K) + weight_perturbation_strength*np.random.rand(K)
    weights /= np.sum(weights)
    # Avoid almost empty classes
    minweight = min(0.005,(K-1)/(n-1)) # Some minimum weight to avoid empty classes
    weights[np.where(weights < minweight)[0]] = minweight

    ## STEP 2: Draw the assignations of each of the vectors to assign
    y = np.random.choice(K,n,p=weights)

    ## STEP 3: Fill the dataset
    # Pre-allocate memory
    X = np.empty((n,d))
    means = np.empty((K,d))
    covariances = np.empty((K,d,d))

    # Loop over the modes and generate each Gaussian
    for k in range(K):

        # Generate mean for this mode
        successful_mu_generation = False
        while not successful_mu_generation:

            mu_this_mode = _gen_params['separation_scale']*np.random.randn(d)
            if k == 0 or _gen_params['separation_min'] == 0:
                successful_mu_generation = True
            else:
                distance_to_closest_mode = min(np.linalg.norm(mu_this_mode - mu_other) for mu_other in means[:k])
                successful_mu_generation = distance_to_closest_mode > _gen_params['separation_min']

        # Generate covariance for this mode
        scale_variance_this_mode = 10**(np.random.uniform(0,_gen_params['covariance_variability_inter']))
        scale_variance_this_mode *= _gen_params['all_covariance_scaling'] # take into account global scaling
        unscaled_variances_this_mode = 10**(np.random.uniform(0,_gen_params['covariance_variability_intra'],d))
        Sigma_this_mode = scale_variance_this_mode*np.diag(unscaled_variances_this_mode)

        # Rotate if necessary
        # (https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices)
        if not grid_aligned:
            rotate_matrix,_ = np.linalg.qr(np.random.randn(d,d))
            Sigma_this_mode = rotate_matrix@Sigma_this_mode@rotate_matrix.T

        # Save the mean and covariance
        means[k] = mu_this_mode
        covariances[k] = Sigma_this_mode

        # Get the indices we have to fill
        indices_for_this_mode = np.where(y == k)[0]
        nb_samples_in_this_mode = indices_for_this_mode.size

        # Fill the dataset with samples drawn from the current mode

        X[indices_for_this_mode] = np.random.multivariate_normal(mu_this_mode, Sigma_this_mode, nb_samples_in_this_mode)


    ## STEP 4: If needed, normalize the dataset
    if normalize is not None:
        if normalize in ['l_2-unit-ball']:
            maxNorm = np.linalg.norm(X,axis=1).max() + 1e-6 # plus smth to have
        elif normalize in ['l_inf-unit-ball']:
            maxNorm = np.abs(X).max() + 1e-6
        else:
            raise Exception('Unreckognized normalization method ({}). Aborting.'.format(normalize))
        # Normalize by maxNorm
        X /= maxNorm
        means /= maxNorm
        covariances /= maxNorm**2

    ## STEP 5: output
    if output_required == 'dataset':
        out = X
    elif output_required == 'GMM':
        out = (X,(weights,means,covariances))
    elif output_required == 'labels':
        out = (X,y)
    elif output_required == 'all':
        out = (X,y,(weights,means,covariances))
    else:
        raise ValueError('Unrecognized output_required ({})'.format(output_required))
    return out


def generateCirclesDataset(K,n,normalize):
    """
    Generate a synthetic 2-D dataset comprising concentric circles/shells.

    Parameters
    ----------
    K: int, the number of circles modes
    n: int, the number of elements in the dataset (cardinality)
    normalize: string (default=None), if not None describes how to normalize the dataset. Available options:
            - 'l_2-unit-ball': the dataset is scaled in the l_2 unit ball (i.e., all l_2 norms are <= 1)
            - 'l_inf-unit-ball': the dataset is projected in the l_inf unit ball (i.e., all entries are <= 1)


    Returns
    -------
    out:  X: (n,d)-numpy array containing the samples.
    """
    weigths = np.ones(K)/K  # True, ideal weigths (balanced case)
    classSizes = np.ones(K) # Actual samples per class
    # (note: we enforce that weigths is the *actual* proportions in this dataset)

    ## Select number of samples of each mode
    balanced = True # FOR NOW,TODO CHANGE LATER
    if balanced:
        classSizes[:-1] = int(n/K)
        classSizes[-1] = n - (K-1)*int(n/K) # ensure we have exactly n samples in dataset even if n % K != 0
    else:
        minweight = min(0.01,(K-1)/(n-1)) # Some minimum weight to avoid empty classes
        weigths = np.random.uniform(minweight,1,K)
        weigths = weigths/np.sum(weigths) # Normalize
        classSizes[:-1] = (weigths[:-1]*n).astype(int)
        classSizes[-1] = n - np.sum(classSizes[:-1])
    classSizes = classSizes.astype(int)

    ## Initialization
    X = None

    ## Some internal params (TODO allow to give them as optional args? kind of arbitrary!)
    #scale_separation = (5/np.sqrt(d)) # Separation of the Gaussians
    #scale_variance_b = np.array([0.05,0.95])/np.sqrt(d) # Bounds on the scale variance (actually, SD)

    ## Add each mode one by one
    for k in range(K):
        classN = classSizes[k]
        #mu = scale_separation*np.random.randn(d)
        #scale_variance = np.random.uniform(scale_variance_b[0],scale_variance_b[1])
        R = 1+3*np.random.randn(1) # mean
        Rs = R + 0.08*np.random.randn(classN)
        thetas = np.random.uniform(0,2*np.pi,classN)
        x1 = np.expand_dims(np.cos(thetas)*Rs,axis=1)
        x2 = np.expand_dims(np.sin(thetas)*Rs,axis=1)

        newCluster = np.concatenate((x1,x2),axis=1)
        if X is None:
            X = newCluster
        else:
            X = np.append(X,newCluster,axis=0)

    if normalize is not None:
        if normalize in ['l_2-unit-ball']:
            maxNorm = np.linalg.norm(X,axis=1).max() + 1e-6 # plus smth to have
        elif normalize in ['l_inf-unit-ball']:
            maxNorm = np.abs(X).max() + 1e-6
        else:
            raise Exception('Unreckognized normalization method ({}). Aborting.'.format(normalize))
        # Normalize by maxNorm
        X /= maxNorm

    return X


def generateSpiralDataset(n,normalize=None,return_density=False):
    """
    Generate a synthetic 2-D dataset made of a spiral.

    Parameters
    ----------
    n: int, the number of elements in the dataset (cardinality)
    normalize: string (default=None), if not None describes how to normalize the dataset. Available options:
            - 'l_2-unit-ball': the dataset is scaled in the l_2 unit ball (i.e., all l_2 norms are <= 1)
            - 'l_inf-unit-ball': the dataset is projected in the l_inf unit ball (i.e., all entries are <= 1)


    Returns
    -------
    out:  X: (n,d)-numpy array containing the samples.
    """

    ## Initialization
    X = None

    # Spiral parameters
    n_spirals = 1
    min_radius = 0.3
    delta_radius_per_spiral = 1.2
    radius_noise =  0.01

    # parameter
    t = np.random.uniform(0,n_spirals,n)

    Rs = min_radius + delta_radius_per_spiral*t + radius_noise*np.random.randn(n)
    thetas = np.remainder(2*np.pi*t,2*np.pi)
    x1 = np.expand_dims(np.cos(thetas)*Rs,axis=1)
    x2 = np.expand_dims(np.sin(thetas)*Rs,axis=1)

    X = np.concatenate((x1,x2),axis=1)


    maxNorm = 1
    if normalize is not None:
        if normalize in ['l_2-unit-ball']:
            maxNorm = np.linalg.norm(X,axis=1).max() + 1e-6 # plus smth to have no round error
        elif normalize in ['l_inf-unit-ball']:
            maxNorm = np.abs(X).max() + 1e-6
        else:
            raise Exception('Unreckognized normalization method ({}). Aborting.'.format(normalize))
        # Normalize by maxNorm
        X /= maxNorm

    # Compute the density function too
    def pdf(x):
        # Compute polar coordinates TODO SUPPORT FOR N SPIRALS > 1
        x1 = x[0] * maxNorm
        x2 = x[1] * maxNorm
        r = np.sqrt(x1**2+x2**2)
        th = np.arctan2(x2,x1)
        if th<0:
            th += 2*np.pi
        return (1/(2*np.pi)) * (scipy.stats.norm.pdf(r, loc=min_radius + delta_radius_per_spiral*th/(2*np.pi), scale=radius_noise)) / r # First part comes from theta, second from R

    if return_density:
        return (X,pdf)
    return X


def generatedataset_Ksparse(d,K,n,max_radius=1):
    """
    Generate a synthetic dataset of K-sparse vectors in dimension d, with l_2 norm <= max_radius.

    Parameters
    ----------
    d: int, the dataset dimension
    K: int, the sparsity level (vectors have at most K nonzero entries)
    n: int, the number of elements in the dataset (cardinality)
    max_radius: real>0, vectors are drawn uniformy in the l_2 ball of radius max_radius

    Returns
    -------
    X: (n,d)-numpy array containing the samples
    """

    # Random points in a ball
    r = max_radius*(np.random.uniform(0,1,size=n)**(1/K)) # Radius, sqrt for uniform density
    v = np.random.randn(n,d) # Random direction
    X = (v.T*(1/np.linalg.norm(v,axis=1))*r).T

    # Random support (sets to zero the coefficients not in the support)
    for i in range(n):
        X[i,np.random.permutation(d)[K:]] = 0

    return X


def sample_ball(radius, npoints, ndim=200, center=None):
    vec = np.random.randn(npoints, ndim)
    vec /= (np.linalg.norm(vec, axis=1).reshape(-1, 1))
    vec *= (np.random.uniform(0, radius, size=npoints).reshape(-1, 1) ** (1. / ndim))
    if center is None:
        return vec
    else:
        return vec + center
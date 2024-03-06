import numpy as np
from sklearn.mixture import GaussianMixture

def train_gmm(features, n_components=3, covariance_type='full', random_state=None):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    gmm.fit(features)
    return gmm

if __name__ == "__main__":
    # Example features (replace this with your own dataset)
    features = np.random.rand(100, 13)  # Assuming MFCCs or other features
    
    # Train GMM
    gmm = train_gmm(features)
    
    # Print parameters of the trained GMM
    print("Weights:", gmm.weights_)
    print("Means:", gmm.means_)
    print("Covariances:", gmm.covariances_)

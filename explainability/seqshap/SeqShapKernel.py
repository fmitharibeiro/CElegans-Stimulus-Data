import numpy as np
from .segmentation import SeqShapSegmentation
from shap import KernelExplainer

class SeqShapKernel:
    def __init__(self, model, seq_num, dataset_name, background="feat_mean", random_seed=None, f=None):
        """
        Initialize the SeqShapKernel object. This class operates on a single sample.

        Parameters:
        - model: The machine learning model to be explained.
        - background: The background dataset used for reference.
        - random_seed: Random seed for reproducibility (optional).
        - f: A function like (#seqs, #feats) -> #seqs
        """
        self.model = model
        self.background = background
        self.random_seed = random_seed
        self.sequence_number = seq_num
        self.dataset_name = dataset_name
        self.k = 0

        if f:
            self.f = f
        else:
            self.f = lambda x: self.model.predict(x) # Default prediction function

    
    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        self.background = self.compute_background(X, self.background)

        seg = SeqShapSegmentation(self.f, self.sequence_number, self.dataset_name)

        segmented_X = seg(X)
        self.k = segmented_X.shape[0]

        # Eq. 8
        self.phi_f = self.compute_feature_explanations(segmented_X)
        pass

    def compute_background(self, X, method):
        if method == "feat_mean":
            return np.mean(X, axis=0)
        else:
            raise NotImplementedError
    
    def compute_feature_explanations(self, subsequences):
        phi_f = np.zeros(subsequences[0].shape[1])  # Initialize feature-level explanations
        self.phi_seq = []

        # Iterate over each subsequence
        for subseq in subsequences:
            explainer = KernelExplainer(self.f, self.background)
            shap_values = explainer.shap_values(subseq)  # TODO: Check if it works as intended

            # Sum the Shapley values for each feature
            phi_f += np.sum(shap_values, axis=0)
            self.phi_seq.append(np.sum(shap_values, axis=0))

        return phi_f
    
    def compute_subsequence_explanations(self, X_prime):
        K, M = X_prime.shape
        phi = np.zeros((K, M))  # Initialize subsequence-level explanations

        for j in range(M):
            Sj = X_prime[:, j].reshape(-1, 1)  # Extract subsequence Sj
            zj = np.ones((K,))  # zj is a binary vector representing the presence of the jth feature in the subsequence

            # Define hx(zj) and g(zj) using some functions (Equations 9 and 10)
            hx_zj = self.hx_function(X_prime, zj, j)
            g_zj = self.g_function(zj, j)

            # Use KernelSHAP to compute the Shapley values for the jth feature in the subsequence
            explainer = KernelExplainer(self.f, self.background)
            phi_j = explainer.shap_values(Sj, zj, hx_zj, g_zj) # TODO: Only Sj is accepted, the rest may need integration

            # Store the Shapley values for the jth feature
            phi[:, j] = phi_j

        return phi

    def hx_function(self, X_prime, zj, j):
        K, M = X_prime.shape
        hx = np.ones((K, M))  # Initialize hx matrix with ones

        # Replace column j with zj
        hx[:, j] = zj

        # Replace 1s with X_prime[i, j] and 0s with background[j]
        hx = np.where(hx == 1, X_prime, self.background)

        return hx

    def g_function(self, zj, j):
        # Compute phi_seq_0_j using background values
        phi_f_0 = np.mean(self.phi_f, axis=0)  # Mean of feature-level attributions
        phi_seq_0_j = phi_f_0 + np.sum(self.phi_f) - self.phi_f[j]  # phi_seq_0_j calculation

        # Compute g_seq_j
        g_seq_j = phi_seq_0_j + np.sum(self.phi_seq[j] * zj, axis=0)

        return g_seq_j


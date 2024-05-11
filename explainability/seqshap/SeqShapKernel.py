import numpy as np, scipy
from .segmentation import SeqShapSegmentation
from shap import KernelExplainer
from shap.utils._legacy import convert_to_link, convert_to_model, convert_to_instance, match_instance_to_data, match_model_to_data
from .utils import convert_to_data, compute_background

class SeqShapKernel(KernelExplainer):
    def __init__(self, model, data, seq_num, dataset_name, background="feat_mean", random_seed=None, **kwargs):
        """
        Initialize the SeqShapKernel object. This class operates on a single sample.

        Parameters:
        - model: A function like (#seqs, #feats) -> #seqs
        - background: The background dataset used for reference.
        - random_seed: Random seed for reproducibility (optional).
        """
        self.keep_index = kwargs.get("keep_index", False)
        self.link = convert_to_link("identity")
        self.model = convert_to_model(model)
        self.data = convert_to_data(data[seq_num:seq_num+1])
        self.background = background
        self.random_seed = random_seed
        self.sequence_number = seq_num
        self.dataset_name = dataset_name
        self.k = 0

        self.model_null = match_model_to_data(self.model, self.data)
        self.fnull = np.sum((self.model_null.T * self.data.weights).T, 0)

        self.vector_out = True
        self.D = self.fnull.shape[0]

        print(f"D: {self.D}")

    
    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        self.background = compute_background(X, self.background)

        seg = SeqShapSegmentation(lambda x: self.model_null[x], self.sequence_number, self.dataset_name)

        segmented_X = seg(X)
        self.k = segmented_X.shape[0]

        # Eq. 8
        self.phi_f = self.compute_feature_explanations(X)

        print(f"Phi_f: {self.phi_f}")

        pass


    def shap_values(self, X, **kwargs):
        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if scipy.sparse.issparse(X) and not scipy.sparse.isspmatrix_lil(X):
            X = X.tolil()
        assert x_type.endswith(arr_type) or scipy.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type

        # single instance
        if len(X.shape) == 2:
            data = X.reshape((1, X.shape[0], X.shape[1]))
            # if self.keep_index:
            #     data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.shape
            out = np.zeros(s)
            out[:] = explanation
            return out
        
        else:
            raise NotImplementedError
    
    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()
            
        print(f"Feat groups: {self.varyingFeatureGroups}")
        print(f"Data groups: {self.data.groups}")
        print(f"Varying ind: {self.varyingInds}")
        print(f"M: {self.M}")

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)

        # Skipped code here (symbolic tensor)

        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]
        
        else:
            raise NotImplementedError
        
        print(f"Phi: {phi.shape}")

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        return phi

    
    def varying_groups(self, x):
        if not scipy.sparse.issparse(x):
            print(f"Groups size: {self.data.groups_size}")
            varying = np.zeros(self.data.groups_size)
            for i in range(0, self.data.groups_size):
                inds = self.data.groups[i]
                x_group = x[0, :, inds]
                if scipy.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[2] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                # Values vary only if they are considerably different from background
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.background[inds]))
                print(f"Num_mismatches for feature {i+1}: {num_mismatches}")
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            raise NotImplementedError # Check original function
    
    def compute_feature_explanations(self, X):
        phi_f = np.zeros(X.shape[1])  # Initialize feature-level explanations

        for j in range(X.shape[1]):
            # Create an array with background values
            background_filled = np.full_like(X, fill_value=self.background)

            # Replace the corresponding feature with values from X
            background_filled[:, j:j+1] = X[:, j:j+1]

            print(f"Backg: {background_filled[0, :]}")
            print(f"self_backg: {self.background}")
            shap_values = self.shap_values(background_filled)  # TODO: Check if it works as intended

            # Sum the Shapley values for each feature (shap_values are shaped inversely)
            phi_f += np.sum(shap_values, axis=1)

        return phi_f
    
    def compute_subsequence_explanations(self, subsequences):
        self.phi_seq = []
        it = 0

        # Iterate over each subsequence
        for subseq in subsequences:
            # Adjust subseq length by removing rows with NaN values
            subseq_aux = subseq[~np.isnan(subseq).any(axis=1)]

            # Create an array with background values
            background_filled = np.full_like(subsequences[0], fill_value=self.background)

            # Replace the corresponding rows with values from subseq_aux
            background_filled[it:it + subseq_aux.shape[0], :] = subseq_aux

            print(f"Subseq: {subseq.shape}")
            shap_values = self.shap_values(background_filled)  # TODO: Check if it works as intended

            self.phi_seq.append(np.sum(shap_values, axis=0))

            it += subseq.shape[0]

        return self.phi_seq
    
    # def compute_subsequence_explanations(self, X_prime):
    #     K, M = X_prime.shape
    #     phi = np.zeros((K, M))  # Initialize subsequence-level explanations

    #     for j in range(M):
    #         Sj = X_prime[:, j].reshape(-1, 1)  # Extract subsequence Sj
    #         zj = np.ones((K,))  # zj is a binary vector representing the presence of the jth feature in the subsequence

    #         # Define hx(zj) and g(zj) using some functions (Equations 9 and 10)
    #         hx_zj = self.hx_function(X_prime, zj, j)
    #         g_zj = self.g_function(zj, j)

    #         # Use KernelSHAP to compute the Shapley values for the jth feature in the subsequence
    #         phi_j = self.shap_values(Sj, zj, hx_zj, g_zj) # TODO: Only Sj is accepted, the rest may need integration

    #         # Store the Shapley values for the jth feature
    #         phi[:, j] = phi_j

    #     return phi

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


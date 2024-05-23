import numpy as np, scipy, copy, itertools
from tqdm.auto import tqdm
from .segmentation import SeqShapSegmentation
from shap import KernelExplainer
from shap.utils._legacy import convert_to_link, convert_to_model, convert_to_instance, match_instance_to_data, match_model_to_data
from .utils import convert_to_data, compute_background
from .plots import visualize_phi_seq
from scipy.special import binom

class SeqShapKernel(KernelExplainer):
    def __init__(self, model, data, seq_num, feat_num, dataset_name, background="feat_mean", random_seed=None, **kwargs):
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
        self.seq_num = seq_num
        self.feat_num = feat_num
        self.dataset_name = dataset_name
        self.k = 0

        self.model_null = match_model_to_data(self.model, self.data)
        self.fnull = np.sum((self.model_null.T * self.data.weights).T, 0)

        self.vector_out = True
        self.D = self.fnull.shape[0]

        self.N = self.data.data.shape[1]
        self.P = self.data.data.shape[2]

        print(f"D, N, P: {self.D}, {self.N}, {self.P}")

        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    
    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        self.background = compute_background(X, self.background)

        # seg = SeqShapSegmentation(lambda x: self.model_null[0, x], self.seq_num, self.dataset_name)
        seg = SeqShapSegmentation(lambda x: np.mean(x, axis=0), self.seq_num, self.feat_num, self.dataset_name, True)

        segmented_X = seg(X)
        self.k = segmented_X.shape[0]

        # Feature explanations
        self.compute_feature_explanations(X)

        # Subsequence explanations
        self.compute_subsequence_explanations(segmented_X)

        print(f"Phi_f: {self.phi_f}")
        print(f"Phi_seq: {self.phi_seq}")

        seg = SeqShapSegmentation(lambda x: x, self.seq_num, self.feat_num, self.dataset_name, False)
        # TODO: Maybe should use self.fnull instead of self.model_null (weights)
        segmented_out = seg(self.model_null[0])

        # Update phi_seq to shape (num_feats, num_subseqs_input, num_subseqs_output)
        self.phi_seq = seg.reshape_phi_seq(self.phi_seq, segmented_out)

        print(f"Phi_seq: {self.phi_seq}")
        print(f"Phi_seq: {self.phi_seq.shape}")

        # Plot phi_seq, shape: num_feats x num_subseqs x num_events (of output)
        visualize_phi_seq(self.phi_seq, f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num+1}", "phi_seq.html",
                          f"Heatmap of Phi_seq for Sequence {self.seq_num+1}, Output feature {self.feat_num+1}")

        


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

        # subsequence explanations
        elif len(X.shape) == 3:
            # self.data needs to be temporarily changed
            prev_data = copy.deepcopy(self.data)
            self.data.group_names = [str(i) for i in range(X.shape[0])]
            self.data.groups = [np.array([i]) for i in range(len(self.data.group_names))]
            self.data.groups_size = len(self.data.groups)
            data = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))

            explanations = []
            for i in tqdm(range(X.shape[2]), disable=kwargs.get("silent", False)):
                explanations.append(self.explain(data, feature=i))

            self.data = copy.copy(prev_data)

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = np.zeros((X.shape[2], s[0], s[1]))
                for i in range(X.shape[2]):
                    outs[i] = explanations[i]
                return outs
            
            # single-output
            else:
                out = np.zeros((s[0], X.shape[2]))
                for i in range(X.shape[2]):
                    out[:, i] = explanations[i]
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
            if instance.x.ndim == 4:
                # Use candidate feature set
                # j = kwargs.get("feature", None)
                # not_j = list(range(instance.x.shape[3]))
                # not_j.remove(j)

                # # Change instance.x[0, :, :, ~j] to background, where there aren't np.nans
                # for i in range(instance.x.shape[1]):  # Iterate over num_subseqs
                #     subseq = instance.x[0, i]  # Get the subsequence
                #     # Find indices where the feature is not equal to j and there are no NaNs
                #     inds = np.where(~np.isnan(subseq[:, not_j]).any(axis=1))
                #     # Replace the corresponding values with the background
                #     instance.x[0, i, inds][0][:, not_j] = self.background[not_j]

                # background_filled = np.full_like(self.data.data, fill_value=self.background)
                # background_filled[0, :, j:j+1] = self.data.data[0, :, j:j+1]

                model_out = self.model.f(instance.x[0])
                
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
            # self.l1_reg = kwargs.get("l1_reg", "auto")
            self.l1_reg = False

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11
            
            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate(instance.x)

            # weight the different subset sizes
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(instance.x, mask, w, feat=kwargs.get("feature", None))
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w, feat=kwargs.get("feature", None))
                else:
                    break
            
            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)

                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance.x, mask, 1.0, feat=kwargs.get("feature", None))
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance.x, mask, 1.0, feat=kwargs.get("feature", None))
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            # for d in range(self.D):
            phi = self.solve(self.nsamples / self.max_samples, feat=kwargs.get("feature", None))
            # phi[self.varyingInds, d] = vphi
            # phi_var[self.varyingInds, d] = vphi_var
        
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
                if x.ndim == 3:
                    x_group = x[0, :, inds]
                else:
                    x_group = x[0, inds]
                    print(x_group[0])
                if scipy.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[2] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                # Values only vary if they are considerably different from background
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.background[inds if x.ndim == 3 else range(len(self.background))]))
                print(f"Num_mismatches for feature {i+1}: {num_mismatches}")
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            raise NotImplementedError # Check original function
    
    def compute_feature_explanations(self, X):
        self.phi_f = np.zeros(X.shape[1])  # Initialize feature-level explanations

        for j in range(X.shape[1]):
            # Create an array with background values
            background_filled = np.full_like(X, fill_value=self.background)

            # Replace the corresponding feature with values from X
            background_filled[:, j:j+1] = X[:, j:j+1]

            print(f"Backg: {background_filled[0, :]}")
            print(f"self_backg: {self.background}")
            shap_values = self.shap_values(background_filled)

            # Sum the Shapley values for each feature (shap_values are shaped inversely)
            self.phi_f += np.sum(shap_values, axis=1)

        return self.phi_f
    
    def compute_subsequence_explanations(self, subsequences):
        self.phi_seq = np.zeros((subsequences.shape[2], subsequences.shape[0], subsequences.shape[1]))

        self.phi_seq = self.shap_values(subsequences)  # TODO: Check if it works as intended

        # self.phi_seq[idx] = np.sum(shap_values, axis=1)
        print(f"Shap vals: {self.phi_seq.shape}")

        return self.phi_seq
    
    # def compute_subsequence_explanations(self, subsequences):
    #     self.phi_seq = np.zeros((subsequences.shape[0], subsequences.shape[2]))
    #     start = 0

    #     # Iterate over each subsequence
    #     for idx, subseq in enumerate(subsequences):
    #         # Adjust subseq length by removing rows with NaN values
    #         subseq_aux = subseq[~np.isnan(subseq).any(axis=1)]

    #         # Create an array with background values
    #         background_filled = np.full_like(subsequences[0], fill_value=self.background)

    #         # Replace the corresponding rows with values from subseq_aux
    #         background_filled[start:start + subseq_aux.shape[0], :] = subseq_aux

    #         print(f"Subseq: {subseq.shape}")
    #         print(f"Backg: {background_filled[:, 0]}")
    #         print(f"self_backg: {self.background}")
    #         shap_values = self.shap_values(background_filled)  # TODO: Check if it works as intended

    #         self.phi_seq[idx] = np.sum(shap_values, axis=1)

    #         start += subseq.shape[0]

    #     return self.phi_seq
    
    def allocate(self, x):
        if scipy.sparse.issparse(self.data.data):
            raise NotImplementedError # See original code
        else:
            self.synth_data = np.tile(x, (self.nsamples, 1, 1, 1))
            print(f"Synth data 1: {self.synth_data.shape}")
            print(f"Data data: {self.data.data.shape}")
            print(f"NSamples: {self.nsamples}")

        self.maskMatrix = np.zeros((self.nsamples, self.M)) # TODO: Check
        self.kernelWeights = np.zeros(self.nsamples)  # TODO: Check
        self.y = np.zeros((self.nsamples, self.D)) # TODO: Check
        self.ey = np.zeros((self.nsamples, self.D)) # TODO: Check
        self.lastMask = np.zeros(self.nsamples) # TODO: Check
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)
    
    def addsample(self, x, m, w, feat=None):
        offset = self.nsamplesAdded
        if isinstance(self.varyingFeatureGroups, (list,)):
            # for j in range(self.M):
            #     for k in self.varyingFeatureGroups[j]:
            #         if m[j] == 1.0:
            #             self.synth_data[offset:offset+1, k] = x[0, k]
            raise NotImplementedError
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                # for group in groups:
                #     self.synth_data[offset:offset+1, group] = x[0, group]
                raise NotImplementedError
            else:
                not_in_groups = self.varyingFeatureGroups[mask==0]
                feat_changed = feat if feat is not None else range(x.shape[2])
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0]

                # print(f"Groups: {groups}")

                # Turn all events in subsequences not in mask to background (for a given feature)
                for not_in_group in not_in_groups:
                    subseq = evaluation_data[not_in_group]

                    # print(f"Not in group: {not_in_group}")
                    # print(f"Init subseq: {np.argmax(~np.isnan(subseq).any(axis=1))}")

                    # Find the index where the first non-NaN values appear in each row
                    subseq_start = np.argmax(~np.isnan(subseq).any(axis=1))
                    subseq = subseq[~np.isnan(subseq).any(axis=1)]

                    # print(f"Subseq start: {subseq_start}")
                    # print(f"Subseq shape: {subseq.shape}")
                    # print(f"Subseq: {subseq}")

                    evaluation_data[not_in_group, subseq_start:subseq_start+subseq.shape[0], feat_changed] = self.background[feat_changed]
                
                # print(f"Eval data: {evaluation_data}")

                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if scipy.sparse.issparse(x) and not scipy.sparse.issparse(self.synth_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_data[offset:offset+1] = evaluation_data

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded - self.nsamplesRun
        data = self.synth_data[self.nsamplesRun:self.nsamplesAdded, :]
        if self.keep_index:
            raise NotImplementedError # See original code
        
        exec_data = np.zeros((num_to_run, self.N, self.P))
        for i in range(num_to_run):
            # print(f"Data i: {data[i].shape}")
            # print(f"Data i: {data[i]}")
            # Join subsequences into a shape (1, num_events, num_feats)
            # Stack all subsequences along the second axis
            stacked_data = np.concatenate(data[i], axis=0)

            # print(f"Stacked data: {stacked_data.shape}")
            # print(f"Stacked data 2: {stacked_data[~np.isnan(stacked_data).any(axis=1)].shape}")

            # Remove any NaN values
            exec_data[i] = stacked_data[~np.isnan(stacked_data).any(axis=1)]

        modelOut = self.model.f(exec_data)
        
        # Skipped code here (pandas, symbolic tensor)

        self.y[self.nsamplesRun:self.nsamplesAdded, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = self.y[i, :] * self.data.weights

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1
    
    def solve(self, fraction_evaluated, feat=None):
        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        if self.l1_reg == "auto": # TODO: Possibly remove this? (try first with l1_reg='num_features(10)')
            print(
                "l1_reg='auto' is deprecated and in a future version the behavior will change from a "
                "conditional use of AIC to simply a fixed number of top features. "
                "Pass l1_reg='num_features(10)' to opt-in to the new default behaviour."
            )
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            raise NotImplementedError("Implement regularization")

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)



        # solve a weighted least squares equation to estimate phi
        # least squares:
        #     phi = min_w ||W^(1/2) (y - X w)||^2
        # the corresponding normal equation:
        #     (X' W X) phi = X' W y
        # with
        #     X = etmp
        #     W = np.diag(self.kernelWeights)
        #     y = eyAdj2
        #
        # We could just rely on sciki-learn
        #     from sklearn.linear_model import LinearRegression
        #     lm = LinearRegression(fit_intercept=False).fit(etmp, eyAdj2, sample_weight=self.kernelWeights)
        # Under the hood, as of scikit-learn version 1.3, LinearRegression still uses np.linalg.lstsq and
        # there are more performant options. See https://github.com/scikit-learn/scikit-learn/issues/22855.
        phi_zero = np.sum(self.phi_f) - self.phi_f[feat]
        y = self.linkfv(self.ey) - phi_zero # TODO: Is this correct?

        # Calculate the design matrix X using the mask matrix
        X = self.maskMatrix

        # Calculate WX and X'WX
        WX = self.kernelWeights[:, None] * X

        # Solve the normal equation to find phi
        try:
            phi = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            # Handle singular matrix error
            print(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            sqrt_W = np.sqrt(self.kernelWeights)
            phi = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]

        # clean up any rounding errors
        for i in range(self.M):
            for j in range(self.D):
                if np.abs(phi[i, j]) < 1e-10:
                    phi[i, j] = 0

        return phi

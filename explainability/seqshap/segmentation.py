import os
import numpy as np

from .plots import plot_metric, plot_subsequences, plot_derivatives_and_variances

class SeqShapSegmentation:
    def __init__(self, f, seq_num, feat_num, dataset_name, is_input):
        self.f = f
        self.k = 2
        self.segmentation = "derivative" # or "distribution"

        self.dataset_name = dataset_name
        self.seq_num = seq_num
        self.input_dir = "input" if is_input else f"output_feat_{feat_num+1}"
        if is_input:
            self.save_file = f"config/{dataset_name}/SeqSHAP/input/Sequence_{seq_num+1}.npy"
        else:
            self.save_file = f"config/{dataset_name}/SeqSHAP/output/Sequence_{seq_num+1}_feat_{feat_num+1}.npy"
        
        self.m = 1 # Number of considered neighbors
        self.min_window = 1
        self.threshold = 0.001 # TODO: Good threshold? (range percentage)
        self.tolerance = 0.001 # Tolerance threshold for constant assumption

    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        initial_set = X
        if X.shape[0] > 64:
            self.k = 64 # max value of subsequences (2^6)
        else:
            self.k = X.shape[0]

        if os.path.exists(self.save_file):
            return np.load(self.save_file, allow_pickle=True)  # Use allow_pickle=True if the array contains object dtype

        return getattr(self, str(self.segmentation)+"_based_segmentation")(initial_set)
    
    def constant_trimming(self, initial_set):
        split_list = set()
        constant_start = False

        # Constant trimming - remove split points if the ones on right and left are the same as itself, on all variables
        for i in range(1, len(initial_set)-1):
            if not (np.allclose(initial_set[i], initial_set[i-1], atol=self.tolerance) and np.allclose(initial_set[i], initial_set[i+1], atol=self.tolerance)):
                split_list.add(i)
                split_list.add(i+1)
                constant_start = True
            elif constant_start and i-1 in split_list:
                split_list.remove(i-1)
                constant_start = False

        return list(split_list)


    def fdist(self, subsequences1, subsequences2, start_m, start_n, X):
        m = subsequences1.shape[0]
        n = subsequences2.shape[0]

        # Transform arrays into predictions using self.f
        dist1 = self.f(X[range(start_m, start_m + m)])
        dist2 = self.f(X[range(start_n, start_n + n)])

        # Calculate the MMD (using max instead of supremum, since max exists and space is finite)
        mmd = np.max(np.abs(np.mean(dist1, axis=0) - np.mean(dist2, axis=0)))

        return mmd

    def calculate_metric(self, subsequences, temp_split_points, X, m=None):
        m = self.m
        total_distance = 0
        for i in range(len(subsequences) - 1):
            for j in range(max(i - m, 0), min(i + m + 1, len(subsequences))):
                distance = self.fdist(subsequences[i], subsequences[j], temp_split_points[i], temp_split_points[j], X)
                sqrt_product = np.sqrt(len(subsequences[i]) * len(subsequences[j]))
                total_distance += distance / sqrt_product
        return total_distance

    def distribution_based_segmentation(self, initial_set):
        # Initialize variables
        subsequences = [initial_set]
        split_points = set()
        split_points.add(0)
        split_points.add(len(initial_set))
        best_dmax = 0
        best_subs = []
        d_diff_list = []
        countdown = 10
        update_seqs = True
        first_d = 0
        
        split_list = self.constant_trimming(initial_set)
        # split_list = list(range(1, len(initial_set)-1)) # No trimming

        if self.k > len(split_list)+1:
            self.k = len(split_list)+1

        self.threshold = (np.max(initial_set) - np.min(initial_set)) * self.threshold

        print(f"split list: {split_list}")

        # Main loop
        while len(subsequences) < self.k and countdown:
            d_max = 0 if len(subsequences) != 1 else np.inf
            p = 0
            best_subsequences = []

            # Iterate over all potential split points
            for i in split_list:
                if i not in split_points:
                    # Temporarily add point i to split points
                    temp_split_points = sorted(list(split_points.union({i})))
                    
                    # Generate subsequences based on new split points
                    temp_subsequences = [
                        initial_set[temp_split_points[j - 1]:temp_split_points[j]]
                        for j in range(1, len(temp_split_points))
                    ]

                    skip = False
                    for j in range(len(temp_subsequences)):
                        if len(temp_subsequences[j]) < self.min_window:
                            skip = True
                            break
                    
                    if skip:
                        continue

                    # Calculate metric for new subsequences, assuming m=1 (the considered neighbors)
                    d = self.calculate_metric(temp_subsequences, temp_split_points, initial_set)

                    # print(f"d: {d}, point: {i}")
                    
                    if d > d_max and len(subsequences) != 1:
                        d_max = d
                        p = i
                        best_subsequences = temp_subsequences
                    elif d < d_max and len(subsequences) == 1:
                        # Choose minimum on the first iteration
                        d_max = d
                        p = i
                        best_subsequences = temp_subsequences
                        first_d = d
            
            # Update split points and segmented subsequences
            split_points.add(p)
            subsequences = best_subsequences

            # metric = (d_max-best_dmax)/(np.sqrt(2*len(subsequences)))
            # metric = (d_max-best_dmax)*((self.k - len(subsequences))**10/(self.k**10))
            metric = d_max-best_dmax

            print(f"Iteration: {len(subsequences)} / {self.k}, Max d: {d_max}, d diff: {d_max-best_dmax}, d diff/it: {metric}, point: {p}")

            d_diff_list.append(metric)
            

            if d_max >= best_dmax:
                if metric < self.threshold and d_max > first_d and countdown == 10:
                    countdown -= 1
                    update_seqs = False
                    print_split_points = set(split_points)
                    print_split_points.remove(p)
                elif countdown < 10:
                    countdown -= 1
                if update_seqs:
                    best_subs = subsequences
                best_dmax = d_max
            
            elif countdown < 10:
                countdown -= 1
            
            if len(subsequences) == self.k and countdown == 10:
                print_split_points = set(split_points)
        
        print(f"print_split: {print_split_points}")

        # Plot iteration vs d_diff
        plot_metric(d_diff_list, "MMD Growth (Penalized by #Subgroups)",
                    f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num+1}/{self.input_dir}",
                    "mmd_growth.png", y_threshold=self.threshold)
        
        # Plot segmentation
        plot_subsequences(initial_set, print_split_points,
                    f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num+1}/{self.input_dir}",
                    "subsequences.png")

        # Solve varying lengths
        max_size = initial_set.shape[0]

        if len(initial_set.shape) > 1:
            ret = np.zeros((len(best_subs), max_size, initial_set.shape[1]))
        else:
            ret = np.zeros((len(best_subs), max_size))

        # Track the current position to insert sequences with NaN padding
        current_pos = 0

        for i, seq in enumerate(best_subs):
            seq_size = len(seq)

            # Calculate the padding needed before and after the sequence
            padding_before = current_pos
            padding_after = max_size - seq_size - padding_before

            # Pad the sequence with NaNs before and after
            if len(initial_set.shape) > 1:
                padding = ((padding_before, padding_after), (0, 0))
            else:
                padding = ((padding_before, padding_after))
            padded_seq = np.pad(seq, padding, mode='constant', constant_values=np.nan)

            # Update the current position
            current_pos += seq_size

            ret[i] = padded_seq
        
        # Save best_subs to file
        if not os.path.exists(self.save_file[:self.save_file.rfind("/")]):
            os.makedirs(self.save_file[:self.save_file.rfind("/")])
        np.save(self.save_file, ret)

        return ret
    
    def derivative_based_segmentation(self, initial_set, min_variance=0.1):
        """
        Segment initial_set based on the discrete derivative.
        
        Parameters
        ----------
        initial_set : np.ndarray
            The input data to be segmented, shaped (num_events, num_feats).
        min_variance : float
            The minimum variance threshold to stop adding split points.
            
        Returns
        -------
        ret : np.ndarray
            The segmented subsequences of the input data, padded with np.nan for varying lengths.
        split_points : set of int
            The points at which the splits occur.
        """
        # Initialize variables
        if len(initial_set) == 2:
            num_events, _ = initial_set.shape
        else:
            num_events = initial_set.shape
        split_points = set()
        split_points.add(0)
        split_points.add(num_events)
        
        # Compute the discrete derivative
        derivative = np.diff(initial_set, axis=0)
        if len(initial_set) == 2:
            variances = np.var(derivative, axis=1)
        else:
            variances = np.abs(derivative)

        max_variance = np.max(variances)

        # Plots
        plot_derivatives_and_variances(derivative, variances,
                        f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num+1}/{self.input_dir}",
                        "derivatives.png", y_threshold=min_variance*max_variance)
        
        while len(split_points) - 1 < self.k:
            # Identify the points with the highest variance
            # max_variance_idxs = np.where(variances == np.max(variances))[0]

            max_variances = np.max(variances)
            max_variance_idxs = np.where(np.isclose(variances, max_variances))[0]

            # print(f"Max variances idx: {max_variance_idxs}")
            # print(f"Split points: {split_points}")
            
            # If the maximum variance is below the threshold, stop
            if variances[max_variance_idxs[0]] < min_variance * max_variance:
                break
            
            # # Ensure first and last points are present
            # if max_variance_idxs[0] not in split_points:
            #     split_points.add(max_variance_idxs[0])
            # if max_variance_idxs[-1] not in split_points:
            #     split_points.add(max_variance_idxs[-1])
            
            # Select the point that maximizes the length of the subsequences
            best_point = None
            max_length = -1
            for idx in max_variance_idxs:
                split_points_temp = sorted(split_points | {idx})
                lengths = [split_points_temp[i + 1] - split_points_temp[i] for i in range(len(split_points_temp) - 1)]
                min_length = min(lengths)
                if min_length > max_length:
                    max_length = min_length
                    best_point = idx
            
            # Add the best point to split_points
            split_points.add(best_point)
            
            # Add neighboring points if it's a single point variation
            if best_point + 2 < num_events and variances[best_point + 1] < min_variance * max_variance and variances[best_point - 1] < min_variance * max_variance:
                split_points.add(best_point + 2)
            
            # Update variances to ignore already split points
            variances[best_point] = -np.inf
            if best_point + 1 in split_points and best_point + 1 < len(variances):
                variances[best_point + 1] = -np.inf

            print(f"New Split Point: {best_point} {best_point + 2 if best_point + 1 in split_points else ''}")
        
        # Segment the data using split_points
        split_points = sorted(split_points)
        best_subs = [initial_set[split_points[i]:split_points[i + 1]] for i in range(len(split_points) - 1)]

        # Plot segmentation
        plot_subsequences(initial_set, split_points,
            f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num+1}/{self.input_dir}",
            "subsequences.png")
        
        # Solve varying lengths
        max_size = initial_set.shape[0]
        ret = np.zeros((len(best_subs), max_size, initial_set.shape[1]))

        # Track the current position to insert sequences with NaN padding
        current_pos = 0

        for i, seq in enumerate(best_subs):
            seq_size = len(seq)

            # Calculate the padding needed before and after the sequence
            padding_before = current_pos
            padding_after = max_size - seq_size - padding_before

            # Pad the sequence with NaNs before and after
            padding = ((padding_before, padding_after), (0, 0))
            padded_seq = np.pad(seq, padding, mode='constant', constant_values=np.nan)

            # Update the current position
            current_pos += seq_size

            ret[i] = padded_seq

        # Save best_subs to file
        if not os.path.exists(self.save_file[:self.save_file.rfind("/")]):
            os.makedirs(self.save_file[:self.save_file.rfind("/")])
        np.save(self.save_file, ret)
        
        return ret



    def reshape_phi_seq(self, phi_seq, segmented_out):
        num_feats, num_subseqs_input, num_output = phi_seq.shape
        num_subseqs_output = segmented_out.shape[0]

        # Initialize the reshaped array
        reshaped_phi_seq = np.empty((num_feats, num_subseqs_input, num_subseqs_output))

        # Iterate over each feature
        for f in range(num_feats):
            # Iterate over each input subsequence
            for i in range(num_subseqs_input):
                # Iterate over each output subsequence
                for j in range(num_subseqs_output):
                    # Get the values from phi_seq corresponding to non-NaN values in segmented_out[j, :]
                    valid_indices = ~np.isnan(segmented_out[j])
                    valid_values = phi_seq[f, i, valid_indices]

                    # Apply the grouping function (max of absolute value, preserving sign)
                    if valid_values.size > 0:
                        max_abs_index = np.argmax(np.abs(valid_values))
                        reshaped_phi_seq[f, i, j] = valid_values[max_abs_index]
                    else:
                        reshaped_phi_seq[f, i, j] = np.nan

        return reshaped_phi_seq

import os
import numpy as np

from .plots import plot_metric, plot_subsequences

class SeqShapSegmentation:
    def __init__(self, f, seq_num, dataset_name, is_input):
        self.f = f
        self.k = 2

        self.dataset_name = dataset_name
        self.seq_num = seq_num
        self.input_dir = "input" if is_input else "output"
        self.save_file = f"config/{dataset_name}/SeqSHAP/{self.input_dir}/Sequence_{seq_num}.npy"
        
        self.m = 10 # Number of considered neighbors
        self.min_window = 1
        self.threshold = 0.05 # TODO: Good threshold?

    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        initial_set = X
        self.k = int(X.shape[0] / 5) # max value of subsequences

        if os.path.exists(self.save_file):
            return np.load(self.save_file, allow_pickle=True)  # Use allow_pickle=True if the array contains object dtype

        return self.distribution_based_segmentation(initial_set)


    def fdist(self, subsequences1, subsequences2, start_m, start_n, X):
        m = subsequences1.shape[0]
        n = subsequences2.shape[0]

        # Transform arrays into predictions using self.f
        dist1 = self.f(X[range(start_m, start_m + m)])
        dist2 = self.f(X[range(start_n, start_n + n)])

        # Calculate the MMD
        mmd = np.abs(np.mean(dist1) - np.mean(dist2))

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

        # Main loop
        while len(subsequences) < self.k and countdown:
            d_max = 0
            p = 0
            best_subsequences = []

            # Iterate over all potential split points
            for i in range(1, len(initial_set)):
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
                    
                    if d > d_max:
                        d_max = d
                        p = i
                        best_subsequences = temp_subsequences
            
            # Update split points and segmented subsequences
            split_points.add(p)
            subsequences = best_subsequences

            # metric = (d_max-best_dmax)*((self.k - len(subsequences))**10/(self.k**10))
            metric = d_max-best_dmax

            print(f"Iteration: {len(subsequences)} / {self.k}, Max d: {d_max}, d diff: {d_max-best_dmax}, d diff/it: {metric}")

            d_diff_list.append(metric)
            

            if d_max > best_dmax:
                if metric < self.threshold and d_max > 1 / self.min_window and countdown == 10:
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

        # Plot iteration vs d_diff
        plot_metric(d_diff_list, "MMD Growth (Penalized by #Subgroups)",
                    f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num}/{self.input_dir}",
                    "mmd_growth.png", y_threshold=self.threshold)
        
        # Plot segmentation
        plot_subsequences(initial_set, print_split_points,
                    f"plots/{self.dataset_name}/SeqSHAP/Sequence_{self.seq_num}/{self.input_dir}",
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
                    
                    # Apply the grouping function (maximum in this case)
                    reshaped_phi_seq[f, i, j] = np.max(valid_values) if valid_values.size > 0 else np.nan
        
        return reshaped_phi_seq

import os
import numpy as np



class SeqShapSegmentation:
    def __init__(self, f, seq_num, dataset_name):
        self.f = f
        self.k = 2
        self.save_file = f"config/{dataset_name}/SeqSHAP/Sequence_{seq_num}.npy"

    def __call__(self, X):
        ''' X shape: (#events, #feats)
        '''
        initial_set = X
        self.k = X.shape[1] # max value of subsequences

        if os.path.exists(self.save_file):
            return np.load(self.save_file, allow_pickle=True)  # Use allow_pickle=True if the array contains object dtype

        return self.distribution_based_segmentation(initial_set)


    def fdist(self, subsequences1, subsequences2):
        m = len(subsequences1)
        n = len(subsequences2)

        # Transform subsequences into arrays
        s1 = np.zeros((m, subsequences1[0].shape[0]))
        s2 = np.zeros((n, subsequences2[0].shape[0]))

        for i, subseq in enumerate(subsequences1):
            s1[i] = subseq

        for i, subseq in enumerate(subsequences2):
            s2[i] = subseq

        # Transform arrays into predictions using self.f
        predictions1 = self.f(s1)
        predictions2 = self.f(s2)

        # Calculate the MMD
        mmd = np.abs(np.mean(predictions1) - np.mean(predictions2))

        return mmd

    def calculate_metric(self, subsequences, m=1):
        total_distance = 0
        for i in range(len(subsequences) - 1):
            for j in range(max(i - m, 0), min(i + m + 1, len(subsequences))):
                distance = self.fdist(subsequences[i], subsequences[j])
                sqrt_product = np.sqrt(len(subsequences[i]) * len(subsequences[j]))
                total_distance += distance / sqrt_product
        return total_distance

    def distribution_based_segmentation(self, initial_set):
        # Initialize variables
        subsequences = [initial_set]
        split_points = set()
        best_dmax = 0
        best_subs = []

        # Main loop
        while len(subsequences) < self.k:
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
                    
                    # Calculate metric for new subsequences, assuming m=1 (the considered neighbors)
                    d = self.calculate_metric(temp_subsequences)
                    
                    if d > d_max:
                        d_max = d
                        p = i
                        best_subsequences = temp_subsequences
            
            # Update split points and segmented subsequences
            split_points.add(p)
            subsequences = best_subsequences

            if d_max > best_dmax:
                best_dmax = d_max
                best_subs = subsequences
        
        # Save best_subs to file
        np.save(self.save_file, best_subs)

        return best_subs

import numpy as np


class SeqShapKernel:
    def __init__(self, model, background, random_seed=None, f=None):
        """
        Initialize the SeqShapKernel object.

        Parameters:
        - model: The machine learning model to be explained.
        - background: The background dataset used for reference.
        - random_seed: Random seed for reproducibility (optional).
        - f: A function like (#seqs, #feats) -> #seqs
        """
        self.model = model
        self.background = background
        self.random_seed = random_seed

        if f:
            self.f = f
        else:
            self.f = lambda x: self.model.predict(x) # Default prediction function


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

    def distribution_based_segmentation(self, initial_set, num_segments):
        # Initialize variables
        subsequences = [initial_set]
        split_points = set()

        # Main loop
        while len(subsequences) < num_segments:
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

        return subsequences
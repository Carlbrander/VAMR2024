import numpy as np
from scipy.spatial.distance import cdist


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be -1 if there is no database descriptor
    with an SSD < lambda * min(SSD). No elements of matches will be equal except for the -1 elements.
    """
    #pass
    #dists = cdist(query_descriptors.T, database_descriptors.T, 'sqeuclidean')
    #matches = np.argmin(dists, axis=1)
    #dists = dists[np.arange(matches.shape[0]), matches]
    ##get min non-zero dist
    #min_non_zero_dist = np.min(dists[dists > 0])
#
    #
    #matches[dists >= match_lambda * min_non_zero_dist] = -1
#
    ## remove double matches
    #unique_matches = np.ones_like(matches) * -1
    #_, unique_match_idxs = np.unique(matches, return_index=True)
    #unique_matches[unique_match_idxs] = matches[unique_match_idxs]
#
    #return unique_matches

    # Compute SSD between query and database descriptors
    dists = cdist(query_descriptors.T, database_descriptors.T, 'sqeuclidean')
    # Find the indices of the two smallest distances for each query descriptor
    sorted_indices = np.argsort(dists, axis=1)
    if sorted_indices.shape[1] >= 2:
        best_matches = sorted_indices[:, 0]
        second_best_matches = sorted_indices[:, 1]

        # Get the best and second-best distances
        best_dists = dists[np.arange(dists.shape[0]), best_matches]
        second_best_dists = dists[np.arange(dists.shape[0]), second_best_matches]
        
        # Apply Lowe's ratio test
        ratio_test = best_dists < match_lambda * second_best_dists
    else:
        # Handle case when there's only one database descriptor
        best_matches = sorted_indices[:, 0]
        best_dists = dists[:, 0]
        ratio_test = np.ones(best_dists.shape, dtype=bool)
   
    # Initialize matches with -1
    matches = np.full(query_descriptors.shape[1], -1)
    
    # Assign matches that pass the ratio test
    matches[ratio_test] = best_matches[ratio_test]
    
    # Ensure unique matches
    unique_matches = np.full_like(matches, -1)
    _, unique_indices = np.unique(matches[matches != -1], return_index=True)
    valid_indices = np.where(matches != -1)[0][unique_indices]
    unique_matches[valid_indices] = matches[valid_indices]
    
    return unique_matches

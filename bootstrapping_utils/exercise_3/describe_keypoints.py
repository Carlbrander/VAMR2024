import numpy as np


#def describeKeypoints(img, keypoints, r):
#    """
#    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
#    coordinates. r is the patch "radius".
#    """
#    pass
#    N = keypoints.shape[1]
#    descriptors = np.zeros([(2*r+1)**2, N])
#    print("image.shape", img.shape)
#    print("r:   ", r)
#    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)
#
#    for i in range(N):
#        kp = keypoints[:, i].astype(np.int32) + r
#        
#        descriptors[:, i] = padded[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)].flatten()
#
#    return descriptors


def describeKeypoints(img, keypoints, r):
    N = keypoints.shape[1]
    descriptors = np.zeros([(2*r+1)**2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)
    pad_shape = padded.shape

    for i in range(N):
        # Swap coordinate order to match image indexing
        kp = keypoints[::-1, i].astype(np.int32) + r

        #switch x and y
        kp = np.flip(kp)
        
        # Bounds checking
        if (kp[0] - r < 0 or kp[0] + r + 1 > pad_shape[0] or 
            kp[1] - r < 0 or kp[1] + r + 1 > pad_shape[1]):
            continue  # Skip this keypoint
        
        patch = padded[
            (kp[0] - r):(kp[0] + r + 1), 
            (kp[1] - r):(kp[1] + r + 1)
        ]
        
        # Verify patch shape before flattening
        if patch.size == 0:
            print(f"Empty patch for keypoint {i}")
            continue
        
        descriptors[:, i] = patch.flatten()

    return descriptors



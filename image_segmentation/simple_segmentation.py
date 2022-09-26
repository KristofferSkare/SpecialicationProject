import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sklearn.decomposition
import sklearn.cluster
import scipy.signal
import cv2


def single_channel_gradient(image, channel=0):
    image = image[:,:,channel]
    X_kernel = np.array([[1,
    0,-1],[2,0,-2],[1,0,-1]])
    Y_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    return scipy.signal.convolve2d(image, X_kernel, mode='same'), scipy.signal.convolve2d(image, Y_kernel, mode='same')

def color_in_range(image, color1, color2):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, color1, color2)/255
    return mask

def filter_out_isolated_pixels(image):
    isolated_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    isolation = scipy.signal.convolve2d(image, isolated_filter, mode='same')
    black_isolated = np.where(isolation == -8)
    white_isolated = np.where(isolation == 8)
    image[black_isolated] = 1
    image[white_isolated] = 0
    return image

def filter_small_regions_dbscan(mask, region_threshold=10, max_iterations=2):
    filtered_mask = mask
 
    dbscan = sklearn.cluster.DBSCAN(p=2, eps=1, min_samples=1)
  
    # Must run several times to remove regions that can have been created by the removal of smaller regions last iterations
    # I think 2 iterations are enough, but not sure
    i=0
    while i < max_iterations:
        white_pixels = np.where(filtered_mask==1)
        black_pixels = np.where(filtered_mask==0)
        white_clusters = dbscan.fit_predict(np.array(white_pixels).T)
        black_clusters = dbscan.fit_predict(np.array(black_pixels).T)
        invalid_white_regions = np.where(np.bincount(white_clusters) < region_threshold)[0]
        invalid_black_regions = np.where(np.bincount(black_clusters) < region_threshold)[0]

        if len(invalid_white_regions) == 0 and len(invalid_black_regions) == 0:
            break

        for c in invalid_white_regions:
            filtered_mask[white_pixels[0][white_clusters==c], white_pixels[1][white_clusters==c]] = 0

        for c in invalid_black_regions:
            filtered_mask[black_pixels[0][black_clusters==c], black_pixels[1][black_clusters==c]] = 1
        
        i+=1
    return filtered_mask

def segment_green(image, low_hsv=(32, 0.1*255, 0.1*255), high_hsv = (80, 255, 255)):
    # This is probably the best one yet
    mask = color_in_range(image, low_hsv, high_hsv)
    filtered_mask = filter_small_regions_dbscan(mask, region_threshold=mask.size*0.0005) 

    filtered_mask.resize((*mask.shape, 1))
    return (image*filtered_mask).astype(np.int32), filtered_mask    



def color_abs_gradient(image):
    gradients = np.zeros((*image.shape[:-1], 6))
    for i in range(3):
        gradients[:,:,i*2], gradients[:,:,i*2 +1] = single_channel_gradient(image, channel=i)
    return np.abs(gradients).mean(axis=2)
   

def color_rms_gradient(image):
    gradients = np.zeros((*image.shape[:-1], 6))
    for i in range(3):
        gradients[:,:,i*2], gradients[:,:,i*2 +1] = single_channel_gradient(image, channel=i)
    return np.linalg.norm(gradients, axis=2)

def relative_thresholding(image, threshold=0.5, color=np.array([0,1,0])):
    image = image/np.linalg.norm(image)
    color = color/np.linalg.norm(color)
    color_proportion = np.dot(image, color) / np.sum(image, axis=2)

    mask = color_proportion > threshold
    return mask

def color_structure_tensor(image):
    Rx, Ry = single_channel_gradient(image, channel=0)
    Gx, Gy = single_channel_gradient(image, channel=1)
    Bx, By = single_channel_gradient(image, channel=2)
    S = np.zeros((image.shape[0], image.shape[1], 2, 2))
    S[:,:,0,0] = Rx**2 + Gx**2 + Bx**2
    S[:,:,0,1] = Rx*Ry + Gx*Gy + Bx*By
    S[:,:,1,0] = S[:,:,0,1]
    S[:,:,1,1] = Ry**2 + Gy**2 + By**2
    return S

def color_harris_image(image, kappa = 0.1):
    S = color_structure_tensor(image)
    det = S[:,:,0,0]*S[:,:,1,1] - S[:,:,0,1]**2
    trace = S[:,:,0,0] + S[:,:,1,1]
    R = det - kappa*trace**2
    return R

def color_segmentation(image, threshold=0.5, color=np.array([0,1,0])):
    mask = relative_thresholding(image, threshold, color)
    segmented_image = np.zeros(image.shape)
    segmented_image[mask] = image[mask]
    return segmented_image

def add_neighbor_features(X):
    n = X.shape[3]
    X_ = np.zeros((X.shape[0], X.shape[1], X.shape[2], n*9))
    rolls = [0,-1,1]
    for i in [0,1,2]:
        for j in [0,1,2]:      
            X_[:,:,:,n*(3*i+j):n*(3*i+j+1)] = np.roll(np.roll(X, rolls[i], axis=1), rolls[j], axis=2)
    return X_

def relative_image(X):
    return X/np.sum(X, axis=3, keepdims=True)

def add_relative_features(X):
    #return X/np.sum(X, axis=3, keepdims=True)
    X_ = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3]*2))
    X_[:,:,:,0:X.shape[3]] = X/255
    X_[:,:,:,X.shape[3]:] = relative_image(X)
    return X_
   

def PCA_clustering(X, num_components=3):
    if len(X.shape) == 3:
        X_ = X
    else:
        scale = np.std(X, axis=0)
        mean = np.mean(X, axis=0)
        X_ = (X - mean) /scale 
    pca = sklearn.decomposition.PCA(n_components=num_components)

    Y = pca.fit_transform(X_.reshape(-1, X_.shape[-1]))
    clust_alg = sklearn.cluster.KMeans(n_clusters=4)
    clusters = clust_alg.fit_predict(Y)
    clustered_data = clusters.reshape(X.shape[:-1])
    min_c = np.min(clustered_data)
    max_c = np.max(clustered_data)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(X[:,:,:3])
    ax[1].imshow((clustered_data- min_c)/(max_c-min_c))
    plt.show()



def load_data(folder, shape=(256,256)):
    images = []
    for file in os.listdir(folder):
        img = Image.open(os.path.join(folder, file)).convert('RGB').resize(shape)
        img = np.array(img)
        images.append(img)
    return np.array(images)

def mean_shift_clustering(image):
    # This is very slow
    cluster_alg = sklearn.cluster.MeanShift()
    X = image.reshape(-1, image.shape[-1])
    clusters = cluster_alg.fit_predict(X)
    cluster_colors = cluster_alg.cluster_centers_/np.max(cluster_alg.cluster_centers_)
    print(cluster_colors)
    clusters_colored = cluster_colors[clusters]
    clustered_image = clusters_colored.reshape(image.shape)
    return clustered_image

def shadow_remove(img):
    # Does not seem to work well: https://medium.com/arnekt-ai/shadow-removal-with-open-cv-71e030eadaf5
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


if __name__ == "__main__":

    X = load_data("./data/plants", shape=(150,150))
    index = 3

    image = X[index]
    
    # This is probably the best one yet
    segmented_image, mask = segment_green(image)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(mask, vmin=0, vmax=1, cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(segmented_image)
    plt.show()
   
  
    '''
    # Harris detector
    image = color_harris_image(X[index], kappa=0.15)
    edge = (image < np.min(image)*0.0001).astype(np.float)
    corner = (image > np.max(image)*0.0001).astype(np.float)
    plt.subplot(1,2,1)
    plt.imshow(edge, cmap='gray', vmin=np.min(edge), vmax=np.max(edge))
    plt.subplot(1,2,2)
    plt.imshow(corner, cmap='gray', vmin=np.min(corner), vmax=np.max(corner))
    plt.show()
    '''

    '''
    # Gradient edge detection
    abs_grad = color_abs_gradient(X[index])
    rms_grad = color_rms_gradient(X[index])
    plt.subplot(1,2,1)
    plt.imshow(abs_grad, cmap='gray', vmin=np.min(abs_grad), vmax=np.max(abs_grad))
    plt.subplot(1,2,2)
    plt.imshow(rms_grad, cmap='gray', vmin=np.min(rms_grad), vmax=np.max(rms_grad))
    plt.show()
    '''
    #X_added_features = add_relative_features(X)
    #X_added_features = add_neighbor_features(X_added_features)
    #PCA_clustering(X_added_features[index], num_components=10)
    # mask = relative_thresholding(X[index], threshold=0.6, color=np.array([120/255,159/255,80/255]))
    
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(X[index])
    # ax[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    # ax[2].imshow(X[index]*mask.reshape(X[index].shape[0], X[index].shape[1], 1))
    # plt.show()
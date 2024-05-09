import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure, img_as_float
from sklearn.decomposition import PCA


from helper_files.displays import get_class_examples
'''
Functions available:
- HOG
- PCA
'''

def HOG(dset, label_dict, orientations=4, pixels_per_cell=(20,20), cells_per_block=(2,2), plot_hog=True, plot_feature_vector = False):
    '''
    Uses skimage.feature.hog to
    :param dset: pass dataset (e.g. train that contains ['image'] and ['label]'
    :param label_dict: global label dictionary
    :param plot: to provide visualization of hog features and
    :return: hog features shape (n_features, 1), list of hog visualizations of length n_images
    '''
    features = []
    h_list = []

    # get hogs
    # Note: hog can only handle 1 color channel. Take last one if using color, else None for grayscale
    # source: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html
    if len(dset[0]['image'].shape) > 2:
        channel_axis = -1
    else:
        channel_axis = None

    for img in dset['image']:
        f, h = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, channel_axis=channel_axis)
        # f, h = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1,1), visualize=True, channel_axis=channel_axis)

        features.append(f)
        h_list.append(h)
    print('Completed HOG calc')

    if plot_hog:
        samples = get_class_examples(dset, label_dict, num_samples=1)
        print('Retrieved samples')

        # get number of classes
        N = len(dset.unique('label'))
        fig, ax = plt.subplots(nrows=2, ncols=N, figsize=(20, 10))

        # plot one example image and HOG visualization per class
        i = -1  # for dynamic plotting
        for class_, sample_idx in samples.items():
            i += 1
            # plot original image
            ax[0, i].imshow(dset['image'][sample_idx[0]], cmap='gray')
            ax[0, i].axis('off')
            ax[0, i].set_title('{}'.format(label_dict[class_]))

            # plot hog
            hog_img = exposure.rescale_intensity(h_list[sample_idx[0]],
                                                 in_range=(0, 10))
            ax[1, i].imshow(hog_img, cmap='gray')
            ax[1, i].axis('off')

        plt.tight_layout()
        plt.show()
    if plot_feature_vector:
        # Plot hog feature vector
        plt.figure(figsize=(10, 5))
        for idx in samples.values():
            plt.plot(features[idx[0]], alpha=0.25)
        plt.title("HOG Feature Vector Plot")
        plt.xlabel("Feature Position Index")
        plt.ylabel("Feature Value")
        plt.legend(label_dict.values())
        plt.show()
    return features, h_list


def pca_explained_variance(feature_vector, title="", n_components=None, plot=True,
                           details=False):
    '''
    Code Source: https://saturncloud.io/blog/what-is-sklearn-pca-explained-variance-and-explained-variance-ratio-difference/

    :param feature_vector: feature vector of shape (n_samples, n_features)
    :param n_components: optional number of components to consider. Defaults to min(n_samples, n_features)
    :param plot: visualization
    :param details: print statements for detailed view
    :return: pca object, number of components to reach 90% explained variance,
             and the index list corresponding to the components responsible for 90% explained variance
    '''
    features = np.array(feature_vector)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit_transform(features)

    # Explained variance:
    # how much of the total variance in the original dataset is explained by each principal component.
    # The explained variance of a principal component is equal to the eigenvalue associated with that component.
    explained_variance = pca.explained_variance_
    total_explained_variance = explained_variance.sum()

    # Explained variance ratio:
    # Proportion of the total variance in the original dataset that is explained by each principal component.
    # The explained variance ratio of a principal component is equal to the
    # ratio of its eigenvalue to the sum of the eigenvalues of all the principal components.
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance_ratio = explained_variance_ratio.sum()

    #
    min_components_90 = np.where(np.cumsum(explained_variance_ratio) <= 0.9)[0][
                            -1] + 2 # 1 to ensure we are above 90 percent, 1 for correct indexing


    # Print results
    if details:
        # print(f"Explained Variance:\n{explained_variance}")
        print(f"Total Explained Variance: {total_explained_variance:.4f}")
        # print(f"\nExplained Variance Ratio:\n{explained_variance_ratio}")
        print(
            f"Total Explained Variance Ratio: {total_explained_variance_ratio:.4f}")
        print(f"Original Number of Components: {features.shape}")
        print(
            f"Number of principal components for 90% explained variance: {min_components_90}")

    # Plot explained variance ratio
    if plot:
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        plt.plot(cumulative_variance_ratio, marker='.',
                 label='Feature Vector Components')
        plt.axhline(y=0.9, color='k', linestyle='--')
        plt.axvline(x=min_components_90-1, color='k', linestyle='-',
                    label=f'{min_components_90} components for 90% Explained Variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.yticks(np.linspace(0, 1, num=11))
        plt.title(f'{title} PCA')
        plt.legend(loc='lower right')
        plt.show()

    return pca, min_components_90

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float
from sklearn.manifold import TSNE

'''
get_class_examples
plot_example_images
dataset_distribution_plot
TSNE
'''

def get_class_examples(dset, label_dict, num_samples, randomize=True):
  '''
  :param dset: dataset
  :param label_dict: global label_dictionary
  :param num_samples: number of examples to return as a list of indexes
  :return: Dictionary {class label number: [index of samples]}
  '''
  idx_labels = {}
  labels = dset['label']
  for i in label_dict.keys():
      idx_labels[i] = list(np.where(labels == i)[0][:num_samples])

      if randomize:
          all_choices = np.where(labels == i)[0]
          idx_labels[i] = list(np.random.choice(all_choices, num_samples))

  return idx_labels


def plot_example_images(dset, label_dict, num_samples, randomize=True):
  '''
  :param dset: full dataset (includes labels and images)
  :param label_dict: the global label_dict created during dataset load
  :param num_samples: number of example images
  :param randomize: whether to take first num_samples or random num_samples
  :return: n classes by m samples grid of images
  '''
  idx_labels = get_class_examples(dset, label_dict, num_samples=num_samples,
                                  randomize=randomize)

  # modify plotting from previous cell, use enumerate to have an i value for row, col of axs
  fig, axs = plt.subplots(nrows=len(label_dict), ncols=num_samples,
                          figsize=(20, 20))
  row = -1
  for key, value in idx_labels.items():
      row += 1

      for i, val in enumerate(value):
          image = dset[int(val)]['image']
          label = dset[int(val)]['label']
          label_name = label_dict[label]

          axs[row, i].imshow(image, cmap='gray')
          axs[row, i].set_title(label_name)
          axs[row, i].axis('off')

  fig.suptitle(f'Images of Each Class', fontsize=30)
  plt.subplots_adjust(top=0.92)
  plt.show()

def dataset_distribution_plot(train, test, val, label_dict):
  '''
  :param train: transformed training dataset
  :param test: transformed test dataset
  :param val: transformed validation dataset
  :param label_dict: global label dictionary
  :return: None. Plotting distribution of each dataset in datasetDict
  '''
  num_classes = len(label_dict.keys())
  ind = np.arange(num_classes)

  plt.figure(figsize=(10, 5))
  width = 0.2

  plt.bar(ind, pd.Series(train['label']).value_counts().sort_index(), width,
          label='Train')
  plt.bar(ind + width, pd.Series(val['label']).value_counts().sort_index(),
          width, label='Val')
  plt.bar(ind + width * 2,
          pd.Series(test['label']).value_counts().sort_index(), width,
          label='Test')

  plt.xlabel('Labels')
  plt.ylabel('Number of Examples')
  plt.title('Train Val Test Distribution')

  plt.xticks(ind + width / 3, label_dict.values())

  plt.legend(loc='best')
  plt.show()

  pass


def tsne_plot(feature_set, targets, label_dict, title=""):
    if isinstance(feature_set, list):
        feature_set = np.array(feature_set)

    # create TSNE
    tsne = TSNE(n_components=2, random_state=0)
    X = tsne.fit_transform(feature_set)  # 2D representation
    y = [0, 1, 3, 5, 6, 7, 8]#range(len(label_dict))

    # visualize
    plt.figure(figsize=(8, 8))
    viridis = plt.colormaps.get_cmap('viridis')
    colors = np.linspace(0, 1, len(label_dict))
    for i, c, l in zip(y, colors, label_dict.values()):
        # plot points based on known labels
        plt.scatter(X[np.where(targets == i), 0],
                    X[np.where(targets == i), 1],
                    color=viridis(c), label=l)
    plt.legend()
    plt.title(title)
    plt.savefig(f"tsne_plots/tsne_{title}.png", bbox_inches='tight')
    plt.show()


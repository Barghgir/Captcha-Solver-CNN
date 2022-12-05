# Data Preprocessing
# First part defines some constatns and at the second part define function that find all the symbols in CAPTCHA and then load X and Y datasets
FILE_PATH = '/content/drive/MyDrive/dataset/AIMedic/samples/'
import numpy as np
import os

def find_char(file_path: str = FILE_PATH) -> tuple([int, dict]):
    """
      Finding out the characters present in captcha images

      Arguments:
          file_path -- Path of datset file
      Returns:
          num_symbols --  Number of symbols
          characters_dict --  Dictionary of all the characters that used in CAPTCHA
    """

    characters_dict = {}
    for i, img in enumerate(os.listdir(file_path)):
        imgName = img.split('.')[0]
        for character in imgName:
            if character not in characters_dict.keys():
                characters_dict[character] = 1
            else:
                characters_dict[character] += 1
    num_symbols = len(characters_dict)
    return num_symbols, characters_dict

# Creating Testing and Training Data
# This part define a finction that split data to X_train, y_train, X_test and y_test


def split_data(X: np.ndarray, Y: np.ndarray, percentage: float = 0.8, num_Samples: int = 1070) -> tuple([np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]):
    """
      Split dataset to train set and test set for input and output sets

      Arguments:
          percentage -- Percentage of picking data from main dataset
          num_Samples -- number of samples
      Returns:
          X_train --  Input training set
          y_train -- Out put training set
          X_test -- Input testing set
          y_test -- Out put testing set
          num_test -- number of test samples
    """

    num_training = int(num_Samples * percentage)
    num_test = int(num_Samples - num_training)
    X_train = X[:num_training]
    y_train = Y[:, :num_training]
    X_test = X[num_training:]
    y_test = Y[:, num_training:]

    return X_train, y_train, X_test, y_test, num_test



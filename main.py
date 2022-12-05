

if __name__=="__main__":

  # Importing the dataset and libraries
  # This part import all the requirement libraries

  import numpy as np
  import tensorflow as tf
  import cv2
  import os
  from data_preprocessing import find_char, split_data
  from model import define_model
  from utils import display_learning_curves, plot_samplee
  from testing_model import eval_model, testing, num_wrong_pred

  FILE_PATH = '/content/drive/MyDrive/dataset/AIMedic/samples/'

  captchaImgShape = (50, 200, 1)
  numSamples = len(os.listdir(FILE_PATH))
  captchaLength = 5

  # Finding out the characters present in captcha images

  numSymbols, captchaCharacters = find_char(file_path=FILE_PATH)

  X = np.zeros((numSamples, 50, 200, 1))
  y = np.zeros((captchaLength, numSamples, numSymbols))
  for i, img in enumerate(os.listdir(FILE_PATH)):
    captchaImg = cv2.imread(FILE_PATH + img, cv2.IMREAD_GRAYSCALE)
    captchaImg = captchaImg / 255.0
    captchaImg = np.reshape(captchaImg, captchaImgShape)
    X[i] = captchaImg
    currentName = np.zeros((captchaLength, numSymbols))
    captchaName = img.split('.')[0]
    for j, character in enumerate(captchaName):
      currentName[j, list(captchaCharacters.keys()).index(character)] = 1
    y[:, i] = currentName

  X_train, y_train, X_test, y_test, numTestingSamples = split_data(X=X, Y=y, percentage=0.8, num_Samples=numSamples)

  # define the model
  my_model = define_model(captchaImgShape)

  tf.keras.utils.plot_model(my_model, show_shapes=True)

  history = my_model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], epochs=100,
                         batch_size=32)

  display_learning_curves(history, detail=True)

  model_eval, char_1_acc, char_2_acc, char_3_acc, char_4_acc, char_5_acc = eval_model(model=my_model, xy=(X_test, y_test))
  print("\n-------------------------------------------------------------------------\n")
  print(f'Character - 1 Accuracy : {char_1_acc * 100} %')
  print(f'Character - 2 Accuracy : {char_2_acc * 100} %')
  print(f'Character - 3 Accuracy : {char_3_acc * 100} %')
  print(f'Character - 4 Accuracy : {char_4_acc * 100} %')
  print(f'Character - 5 Accuracy : {char_5_acc * 100} %')
  print("\n-------------------------------------------------------------------------\n")
  predictedCaptchaText, testCaptchaText = testing(model=my_model, cap_char_dict=captchaCharacters, xy=(X_test, y_test),
                                                  num_Symbols=numSymbols, num_TestingSamples=numTestingSamples)

  correctPredictions, wrongPredictions = num_wrong_pred(predicted_CaptchaText=predictedCaptchaText,
                                                        test_CaptchaText=testCaptchaText)
  
  print("\n-------------------------------------------------------------------------\n")
  print(f'Correct Predictions : {correctPredictions}\nWrong Predictions : {wrongPredictions}')
  print(f'Accuracy : {correctPredictions * 100 / (correctPredictions + wrongPredictions)} %')
  print("\n-------------------------------------------------------------------------\n")
  print("\n-------------------------------------------------------------------------\n")
  plot_samplee(predicted_CaptchaText=predictedCaptchaText, xy=(X_test, y_test), test_CaptchaText=testCaptchaText, num_TestingSamples=numTestingSamples)

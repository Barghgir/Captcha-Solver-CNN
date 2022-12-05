
# Testing the Model
# This part Evaluates the model and test it, enhance, and show some examples
from model import define_model
import keras
import numpy as np


def eval_model(model: keras.engine.functional.Functional = None, xy: tuple=()) -> tuple([list, float, float, float, float]):
    """
      Evaluate the model and return the accuracy of the evaluated model

      Arguments:
          model -- The model
          xy -- Tuple of x and y test

      Returns:
          model_eval --  A list that shows loss and accuracy of each character in CAPTCHA
          model_eval[6 to 10] -- accuracy of each character in CAPTCHA

    """

    X_test, y_test = xy
    model_eval = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

    return model_eval, model_eval[6], model_eval[7], model_eval[8], model_eval[9], model_eval[10]


def testing(model:keras.engine.functional.Functional=None, cap_char_dict: dict={}, xy:tuple=(), num_Symbols:int=19,
            num_TestingSamples:int=254) -> tuple([list, list]):
    """
      Test the model and shows the predict and test CAPTCHAs

      Arguments:
          model -- The model
          cap_char_dict -- Dictionary of all the characters in CAPTCHAs
          xy -- Tuple of x and y test
          num_Symbols -- Number of symbols
          num_TestingSamples -- Number of testing samples


      Returns:
          predicted_CaptchaText --  list of predicted CAPTCHAs
          test_CaptchaText -- list of tested CAPTCHAs

    """

    characters = list(cap_char_dict.keys())
    predicted_CaptchaText = []
    test_CaptchaText = []
    X_test, y_test = xy

    for i, xt in enumerate(X_test):
        xt = np.reshape(X_test[i], (1, 50, 200, 1))
        yp = model.predict(xt)
        yp = np.reshape(yp, (5, num_Symbols))
        pct = ''.join([characters[np.argmax(i)] for i in yp])
        predicted_CaptchaText.append(pct)

    for i in range(0, num_TestingSamples):
        tct = ''.join([characters[i] for i in (np.argmax(y_test[:, i], axis=1))])
        test_CaptchaText.append(tct)

    return predicted_CaptchaText, test_CaptchaText


def num_wrong_pred(predicted_CaptchaText: list = [],
                   test_CaptchaText: list = []) -> tuple([int, int]):
    """
      Show that how many of CAPTCHAs predicted wrong or True

      Arguments:
          predicted_CaptchaText --  list of predicted CAPTCHAs
          test_CaptchaText -- list of tested CAPTCHAs

      Returns:
          correct_Predictions --  Number of correct Predictions
          wrong_Predictions -- Number of wrong Predictions

    """
    correct_Predictions = 0
    wrong_Predictions = 0
    for i in range(0, len(predicted_CaptchaText)):
        if predicted_CaptchaText[i] == test_CaptchaText[i]:
            correct_Predictions += 1
        else:
            wrong_Predictions += 1

    return correct_Predictions, wrong_Predictions






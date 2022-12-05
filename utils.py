# plot learning curve
import keras
import matplotlib.pyplot as plt
import numpy as np



def display_learning_curves(history: keras.callbacks.History, detail: bool = True) -> None:
    """
      Display the learning curve for the model

      Arguments:
          history -- History of model
          detail -- If became True, show the main training loss

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    legends = ["train"]
    if not detail:
        ax1.plot(history.history["loss"])

    for i in range(1, 6):
        ax1.plot(history.history['character' + str(i) + '_loss'])
        ax2.plot(history.history['character' + str(i) + '_accuracy'])
        legends.append("char_" + str(i))

    ax1.legend(legends, loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    legends.pop(0)
    ax2.legend(legends, loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.savefig(fname="learning_curve.png")
    plt.show()


def plot_samplee(predicted_CaptchaText: list = [], xy:tuple=(),test_CaptchaText: list = [], num_TestingSamples:int=254) -> None:
    """
      plot number of samples in some pistures

      Arguments:
          predicted_CaptchaText --  list of predicted CAPTCHAs
          xy -- Tuple of x and y test
          test_CaptchaText -- list of tested CAPTCHAs
          num_TestingSamples -- Number of testing samples

    """

    X_test, y_test = xy

    num = int(num_TestingSamples / 2)
    
    for i in range(0, int(num)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(np.reshape(X_test[2 * i], (50, 200)))
        axs[0].set_title(f'Predicted Text : {predicted_CaptchaText[2 * i]} | Actual Text : {test_CaptchaText[2 * i]}')
        axs[1].imshow(np.reshape(X_test[2 * i + 1], (50, 200)))
        axs[1].set_title(
            f'Predicted Text : {predicted_CaptchaText[2 * i + 1]} | Actual Text : {test_CaptchaText[2 * i + 1]}')
        plt.savefig(fname="./pred_images/predictions"+ str(i) + ".png")
    plt.show()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Import sample sine curve (with noise)
data = pd.read_excel(
    os.path.realpath(
        "C:\\Users\\Fhyarnir\\Documents\\pythonMain\\ml\\PredictSineCurve\\sine.xlsx"
    )
)


class optStruct:

    def __init__(self, dataMatIn, classLabels, gamma, sigma):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.gamma = gamma
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], sigma)


def leastSquares(dataMatIn, classLabels, gamma, sigma):
    oS = optStruct(dataMatIn, classLabels, gamma, sigma)

    # Prepare matrix parts
    leftOnes = np.mat(np.ones((oS.m, 1)))  # [[1][1][1][etc]]
    innerMatrix = oS.K + np.identity(oS.m) * (1 / gamma)
    zeroEntry = np.mat(np.zeros((1, 1)))  # [[0]]
    topOnes = leftOnes.T  # [[1 1 1 etc]]

    # Create final matrices
    topPart = np.hstack((zeroEntry, topOnes))
    botPart = np.hstack((leftOnes, innerMatrix))
    matrix = np.vstack((topPart, botPart))
    solution = np.vstack((zeroEntry, oS.labelMat))

    # Calculate bias and alpha values
    b_alpha = matrix.I * solution  # Inverse matrix imprinted on solution vector form
    oS.b = b_alpha[0, 0]
    oS.alphas = b_alpha[1:, 0]

    # Calculate e, which can be used to estimate error weights
    e = oS.alphas / gamma

    return oS.alphas, oS.b, e


def kernelTrans(X, A, sigma):
    m = np.shape(X)[0]
    K = np.mat(np.zeros((m, 1)))
    for j in range(m):
        deltaRow = X[j] - A
        K[j] = deltaRow * deltaRow.T
    K = np.exp(K / (-1 * sigma ** 2))
    return K


def predict(alphas, b, dataMat):
    m = np.shape(dataMat)[0]
    predict_result = np.mat(np.zeros((m, 1)))
    for i in range(m):
        Kx = kernelTrans(dataMat, dataMat[i, :], sigma)
        predict_result[i, 0] = Kx.T * alphas + b
    return predict_result


def predict_singular_points(data, x, y):
    xlocal = np.squeeze(x)
    ylocal = np.squeeze(y)
    rows = np.shape(data)[0]
    y_singular_predictions = np.zeros((rows, 1))
    for i in range(rows):
        zeroToOne = math.modf(data[i, 0])[0]
        y_singular_predictions[i, 0] = np.interp(zeroToOne, xlocal, ylocal)
    return y_singular_predictions


if __name__ == "__main__":
    print("--------------------Load Data------------------------")
    dataMat = np.mat(data.iloc[:, 0]).T
    labelMat = np.mat(data.iloc[:, 1]).T
    print("--------------------Parameter Setup------------------")
    gamma = 0.6
    sigma = 0.3
    print("-------------------Save LSSVM Model-----------------")
    alphas, b, e = leastSquares(dataMat, labelMat, gamma, sigma)
    print("------------------Predict Result------------------ -")
    predict_result = predict(alphas, b, dataMat)

    # Prepare and sort results
    x, y, y_predict = np.array(dataMat), np.array(labelMat), np.array(predict_result)
    x, y, y_predict = zip(*sorted(zip(x, y, y_predict)))

    # Check other new data
    x_new = dataMat.copy() + 2
    alphas_new = alphas.copy()  # Contains "logic" of the curve behavior
    y_new = predict(alphas_new, b, x_new)  # b tends to shift curve up or down
    x_new, y_new = np.array(x_new), np.array(y_new)
    x_new, y_new = zip(*sorted(zip(x_new, y_new)))

    # Example: Predict with interpolation
    test_x = np.zeros((3, 1))
    test_x[0, 0] = 1.0 - x[0]
    test_x[1, 0] = 1.5 - x[0]
    test_x[2, 0] = 2.05 - x[0]
    test_y = predict_singular_points(test_x, x, y_predict)
    test_x, test_y = np.array(test_x), np.array(test_y)
    test_x, test_y = zip(*sorted(zip(test_x, test_y)))

    fig = plt.figure()
    ax = plt.subplot(111)

    # Plot results
    ax.scatter(
        x,
        y,
        marker="o",
        facecolors="none",
        edgecolors="blue",
        label="Training data (N,1)",
    )
    ax.plot(
        x,
        y_predict,
        linestyle="dashed",
        color="red",
        linewidth=3,
        label="Trained model",
    )
    ax.plot(
        x_new,
        y_new,
        linestyle="dashed",
        color="black",
        linewidth=3,
        label="Predict new (N,1)",
    )
    ax.scatter(
        test_x,
        test_y,
        marker="o",
        color="red",
        edgecolors="black",
        s=130,
        linewidths=1.5,
        label="Predict (n,1), n<N",
    )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.10, box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=2)

    plt.show()


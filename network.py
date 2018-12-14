from csvloader import CSVLoader
import numpy as np

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

class HiddenLayer():

    def __init__(self):
        self.x = self.y = self.w = self.b = self.s = self.t = None
        self.formerLayer = None

    def get_layer_sigmoid_gd(self, idx):
        return self.t[idx].dot(self.w)

    def get_sigmoid_gd(self, idx):
        return self.t[idx]

    def update(self):
        if self.formerLayer != None:
            self.x = self.formerLayer.s
        self.y = self.x.dot(self.w) + self.b
        self.s = sigmoid(self.y)
        self.t = self.s * (1 - self.s)


class OutputLayer():

    def __init__(self):
        self.x = self.y = self.w = self.b = None
        self.formerLayer = None

    def get_layer_gd(self, idx):
        return self.w * self.y[idx]

    def get_gd(self, idx):
        return self.y[idx]

    def update(self):
        if self.formerLayer != None:
            self.x = self.formerLayer.s
        self.y = self.x.dot(self.w) + self.b


class Network():

    def __init__(self):
        self.data = self.contact = self.duration = self.other = self.social = self.label = None
        self.l1 = self.l2 = HiddenLayer()
        self.l3 = OutputLayer()
        self.l3.formerLayer = self.l2
        self.l2.formerLayer = self.l1

    def getData(self, path):
        loader = CSVLoader()
        self.data, self.contact, self.duration, self.other, self.social, self.label = loader.getData('bank-additional.csv')

    def train(self, learning_rate=0.01, batch_size=128, epoch=1000, c=1, gamma=0.1):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        self.l1.w = self.l2.w = self.l3.w = np.zeros(x.shape[1])
        self.l1.b = self.l2.b = self.l3.b = 0.
        self.l1.x = x
        for i in range(0, epoch):
            y_pred = self.l3.y
            idx = np.argmax(np.maximum(0, -y_pred * y))
            if y[idx] * y_pred[idx] > 0:
                break
            delta1 = learning_rate * (self.l2.get_layer_sigmoid_gd(idx) * self.l1.get_sigmoid_gd(idx)).dot(self.l3.get_layer_gd(idx))
            delta2 = learning_rate * self.l2.get_sigmoid_gd(idx).dot(self.l3.get_layer_gd(idx))
            delta3 = learning_rate * self.l3.get_gd(idx)
            self.l1.w += delta1 * self.l1.x[idx]
            self.l1.b += delta1
            self.l2.w += delta2 * self.l2.x[idx]
            self.l2.b += delta2
            self.l3.w += delta3 * self.l3.x[idx]
            self.l3.b += delta3
            self.l1.update()
            self.l2.update()
            self.l3.update()

    def predict(self, _x):
        x1 = np.asarray(_x, np.float32)
        x2 = sigmoid(x1.dot(self.l1.w) + self.l1.b)
        x3 = sigmoid(x2.dot(self.l2.w) + self.l2.b)
        y_pred = x3.dot(self.l3.w) + self.l3.b
        return y_pred


if __name__ == '__main__':
    n = Network()
    n.getData('bank-additional.csv')

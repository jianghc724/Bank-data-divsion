from csvloader import CSVLoader
import numpy as np

def sigmoid(x):
    # print(x)
    return 1.0/(1 + np.exp(-x))

class HiddenLayer():

    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        self._b = None
        self.s = None
        self.t = None
        self.formerLayer = None

    def get_layer_sigmoid_gd(self):
        pass

    def get_sigmoid_gd(self):
        pass

    def update(self):
        if self.formerLayer != None:
            self.x = self.formerLayer.s
        self.y = (np.dot(self.w, self.x) + self.b)
        # print("w:", self.w)
        # print("b:", self.b)
        # print("y:", self.y)
        self.s = sigmoid(self.y)
        self.t = self.s * (1 - self.s)


class OutputLayer():

    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        self._b = None
        self.formerLayer = None

    def get_layer_gd(self):
        pass

    def get_gd(self):
        pass

    def update(self):
        if self.formerLayer != None:
            self.x = self.formerLayer.s
        self.y = np.dot(self.w, self.x) + self.b
        # print(self.y)


class Network():

    def __init__(self):
        self.data = None
        self.t_data = None
        self.contact = None
        self.t_contact = None
        self.duration = None
        self.t_duration = None
        self.other = None
        self.t_other = None
        self.social = None
        self.t_social = None
        self.label = None
        self.t_label = None
        self.l1 = HiddenLayer()
        self.l2 = OutputLayer()
        self.l2.formerLayer = self.l1
        self.l1.formerLayer = None

    def getData(self, path):
        loader = CSVLoader()
        self.data, self.contact, self.duration, self.other, self.social, self.label, self.t_data, self.t_contact, self.t_duration, self.t_other, self.t_social, self.t_label = loader.getData(path)

    def train(self, learning_rate=0.01, epoch=1000):
        x = np.asarray(self.data, np.float32).T
        y = np.asarray(self.label, np.float32).T
        self.l1.w = np.zeros((x.shape[0], x.shape[0]))
        self.l2.w = np.zeros((1, x.shape[0]))
        self.l1.b = np.zeros((x.shape[0], x.shape[1]))
        self.l2.b = np.zeros((1, x.shape[1]))
        self.l1._b = np.zeros((x.shape[0], 1))
        self.l2._b = np.zeros((1, 1))
        self.l1.x = x
        for i in range(0, epoch):
            if i % 100 == 0:
                print("Epoch:", i)
            self.l1.update()
            self.l2.update()
            delta2 = self.l2.y - y
            delta1 = np.dot(self.l2.w.T, delta2)
            # print(delta1)
            # print(delta2)
            # print(delta2.shape, self.l1.s.shape)
            self.l1.w -= learning_rate * (np.dot(delta1 * self.l1.t, self.l1.x.T))
            self.l1.b -= learning_rate * np.mean((delta1 * self.l1.t))
            self.l2.w -= learning_rate * np.dot(delta2, self.l1.s.T)
            self.l2.b -= learning_rate * np.mean(delta2)
        self.l1._b = self.l1.b.T[0]
        self.l2._b = self.l2.b.T[0]
        # print(self.l1.b)
        # print(self.l1._b.shape)
        # print(self.l2.b)
        # print(self.l2._b)

    def predict(self, _x):
        self.l1.b = np.tile(self.l1._b, (_x.shape[1], 1)).T
        y1 = np.dot(self.l1.w, _x) + self.l1.b
        s1 = sigmoid(y1)
        self.l2.b = np.tile(self.l2._b, (_x.shape[1], 1)).T
        y_pred = np.dot(self.l2.w, s1) + self.l2.b
        return y_pred

    def test(self):
        n = len(self.t_data)
        correct = 0
        _x = x = np.asarray(self.t_data, np.float32).T
        y = self.predict(_x).T
        for i in range(0, n):
            if y[i] * self.t_label[i] > 0:
                correct += 1
        print("Total:", n)
        print("Correct", correct)
        print("Accuracy:", float(correct) / float(n))


if __name__ == '__main__':
    n = Network()
    n.getData('bank-additional-full.csv')
    n.train()
    n.test()

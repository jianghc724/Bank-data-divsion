from csvloader import CSVLoader
import numpy as np

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

class HiddenLayer():

    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None
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
        self.y = (np.dot(self.w, self.x) + self.b).T
        self.s = sigmoid(self.y)
        self.t = self.s * (1 - self.s)


class OutputLayer():

    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        self.formerLayer = None

    def get_layer_gd(self):
        pass

    def get_gd(self):
        pass

    def update(self):
        if self.formerLayer != None:
            self.x = self.formerLayer.s
        self.y = (np.dot(self.w, self.x) + self.b).T
        print(self.y)


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
        self.data, self.contact, self.duration, self.other, self.social, self.label, self.t_data, self.t_contact, self.t_duration, self.t_other, self.t_social, self.t_label = loader.getData('bank-additional.csv')

    def train(self, learning_rate=0.01, epoch=1000):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        y = y.reshape(x.shape[0], 1)
        self.l1.w = np.zeros((x.shape[0], x.shape[0]))
        self.l2.w = np.zeros((1, x.shape[1]))
        self.l1.b = np.zeros((x.shape[0], x.shape[1]))
        self.l2.b = np.zeros((1, x.shape[0]))
        self.l1.x = x
        for i in range(0, epoch):
            print("Epoch:", i)
            self.l1.update()
            self.l2.update()
            delta2 = self.l2.y.T - y.T
            delta1 = np.dot(delta2.T, self.l2.w)
            #print(delta1.shape, self.l1.t.shape)
            self.l1.w -= learning_rate * (np.dot(delta1 * self.l1.t.T, self.l1.x.T))
            self.l1.b -= learning_rate * (delta1 * self.l1.t.T)
            self.l2.w -= learning_rate * np.dot(delta2, self.l1.y.T)
            self.l2.b -= learning_rate * delta2

    def predict(self, _x):
        pass

    def test(self):
        n = len(self.t_data)
        correct = 0
        for i in range(0, n):
            y = self.predict(self.t_data[i])
            if y * self.t_label[i] > 0:
                correct += 1
        print("Total:", n)
        print("Correct", correct)
        print("Accuracy:", float(correct) / float(n))


if __name__ == '__main__':
    n = Network()
    n.getData('bank-additional.csv')
    n.train()
    n.test()

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
        self.train_data = None
        self.test_data = None
        self.l1 = HiddenLayer()
        self.l2 = OutputLayer()
        self.l2.formerLayer = self.l1
        self.l1.formerLayer = None

    def getData(self, path):
        loader = CSVLoader()
        self.data, self.contact, self.duration, self.other, self.social, self.label, self.t_data, self.t_contact, self.t_duration, self.t_other, self.t_social, self.t_label = loader.getData(path)
        self.train_data = self.data
        self.test_data = self.t_data

    def train(self, learning_rate=0.00001, epoch=10000):
        x = np.asarray(self.train_data, np.float32)
        y = np.asarray(self.label, np.float32)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        # print(idx[0:20])
        x = x[idx].T
        y = y[idx].T
        self.l1.w = np.random.rand(x.shape[0], x.shape[0])
        self.l2.w = np.random.rand(1, x.shape[0])
        self.l1.b = np.random.rand(x.shape[0], x.shape[1])
        self.l2.b = np.random.rand(1, x.shape[1])
        self.l1._b = np.random.rand(x.shape[0], 1)
        self.l2._b = np.random.rand(1, 1)
        self.l1.x = x
        for i in range(0, epoch):
            self.l1.update()
            self.l2.update()
            delta2 = self.l2.y - y
            # print(np.mean(delta2))
            delta1 = np.dot(self.l2.w.T, delta2)
            if i % 1000 == 0:
                print("Epoch:", i)
                # print(delta1)
                print(np.mean(delta2))
            # print(delta2.shape, self.l1.s.shape)
            self.l1.w -= learning_rate * (np.dot(delta1 * self.l1.t, self.l1.x.T))
            self.l1.b -= learning_rate * (delta1 * self.l1.t)
            self.l2.w -= learning_rate * np.dot(delta2, self.l1.s.T)
            self.l2.b -= learning_rate * delta2
        self.l1._b = np.mean(self.l1.b, axis=1)
        self.l2._b = np.mean(self.l2.b)
        print("w1:", self.l1.w)
        print("b1:", self.l1._b)
        print("w2:", self.l2.w)
        print("b2:", self.l2._b)

    def predict(self, _x):
        self.l1.b = np.tile(self.l1._b, (_x.shape[1], 1)).T
        y1 = np.dot(self.l1.w, _x) + self.l1.b
        s1 = sigmoid(y1)
        self.l2.b = np.tile(self.l2._b, (_x.shape[1], 1)).T
        y_pred = np.dot(self.l2.w, s1) + self.l2.b
        return y_pred

    def test(self):
        n = len(self.t_data)
        print(n)
        a = 0
        b = 0
        c = 0
        _x = np.asarray(self.t_data, np.float32).T
        y = self.predict(_x).T
        # print(y)
        # print(self.l2.y)
        for i in range(0, n):
            if y[i] * self.t_label[i] > 0:
                if self.t_label[i] > 0:
                    a += 1
            else:
                if self.t_label[i] > 0:
                    b += 1
                else:
                    c += 1
        print("Precision:", float(a) / float(a + c))
        print("Recall:", float(a) / float(a + b))


if __name__ == '__main__':
    print("network")
    n = Network()
    n.getData('bank-additional.csv')
    n.train()
    n.test()

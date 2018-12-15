from csvloader import CSVLoader
import numpy as np

class SVMHandler():

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
        self.alpha = None
        self.b = None

    def getData(self, path):
        loader = CSVLoader()
        self.data, self.contact, self.duration, self.other, self.social, self.label, self.t_data, self.t_contact, self.t_duration, self.t_other, self.t_social, self.t_label = loader.getData(path)

    def train_gd(self, learning_rate=0.01, epoch=1000, c=1, gamma=0.1):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        k = self.rbf_kernel(x, x, gamma)
        self.alpha = np.zeros(len(x))
        self.b = 0.
        for i in range(0, epoch):
            if i % 100 == 0:
                print("Epoch:", i)
            e = - y * (self.alpha.dot(k) + self.b)
            if np.max(e) < 0:
                continue
            mask = e >= 0
            delta1 = learning_rate * c * y[mask]
            delta2 = learning_rate * 0.5 * self.alpha[mask]
            delta3 = learning_rate * 0.5 * self.alpha
            er = (np.sign(np.sign(e) + 0.1) + 1) * 0.5
            # print(delta2)
            # print(np.diag(k[mask]))
            s = delta3 * np.diag(k) * er
            # print(s.shape)
            # self.alpha += np.sum(delta[..., None] * k[mask], axis=0)
            self.alpha = self.alpha - np.sum(delta2[..., None] * k[mask], axis=0) - s + np.sum(delta1[..., None] * k[mask], axis=0)
            self.b += np.sum(delta1)
            # print(self.alpha, self.b)

    def train_perp(self, learning_rate=0.01, epoch=1000, gamma=0.1):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        k = self.rbf_kernel(x, x, gamma)
        self.alpha = np.zeros(len(x))
        self.b = 0.
        for i in range(0, epoch):
            if i % 100 == 0:
                print("Epoch:", i)
            e = - y * (self.alpha.dot(k) + self.b)
            if np.max(e) < 0:
                continue
            mask = e >= 0
            delta = learning_rate * y[mask]
            self.alpha += np.sum(delta[..., None] * k[mask], axis=0)
            self.b += np.sum(delta)

    def train_smo(self, epoch=1000, c=1, gamma=1):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        k = self.pol_kernel(x, x, gamma / x.shape[1])
        g = self.get_gram(x)
        # print(g)
        self.alpha = np.zeros(len(x))
        # print(x.shape, y.shape, k.shape, self.alpha.shape)
        self.b = 0.
        for i in range(0, epoch):
            con1 = self.alpha > 0
            con2 = self.alpha < c
            y_pred = np.dot(self.alpha.T, k) + self.b
            # print(y_pred, y)
            err1 = y * y_pred - 1
            err2 = err1.copy()
            err3 = err1.copy()
            err1[(con1 & (err1 <= 0)) | (~con1 & (err1 > 0))] = 0
            err2[((~con1 | ~con2) & (err2 != 0)) | ((con1 & con2) & (err2 == 0))] = 0
            err3[(con2 & (err3 >= 0)) | (~con2 & (err3 < 0))] = 0
            err = err1 ** 2 + err2 ** 2 + err3 ** 2
            idx1 = np.argmax(err)
            idx2 = np.random.randint(len(y))
            eta = g[idx1][idx1] + g[idx2][idx2] - 2 * g[idx1][idx2]
            while ((idx2 == idx1) or (eta == 0)):
                idx2 = np.random.randint(len(y))
                eta = g[idx1][idx1] + g[idx2][idx2] - 2 * g[idx1][idx2]
            ee1 = y_pred[idx1] - y[idx1]
            ee2 = y_pred[idx2] - y[idx2]
            alpha1_old = self.alpha[idx1]
            alpha2_old = self.alpha[idx2]
            alpha2_new = alpha2_old + (y[idx2] * (ee1 - ee2)) / eta
            lower_bound = 0
            upper_bound = c
            if y[idx1] != y[idx2]:
                lower_bound = max(0., alpha2_old - alpha1_old)
                upper_bound = min(c, c + alpha2_old - alpha1_old)
            else:
                lower_bound = max(0., alpha1_old + alpha2_old - c)
                upper_bound = min(c, alpha1_old + alpha2_old)
            if alpha2_new > upper_bound:
                alpha2_new = upper_bound
            elif alpha2_new < lower_bound:
                alpha2_new = lower_bound
            alpha1_new = alpha1_old - y[idx1] * y[idx2] * (alpha2_new - alpha2_old)
            self.alpha[idx1] = alpha1_new
            self.alpha[idx2] = alpha2_new
            b1 = -ee1 - y[idx1] * g[idx1][idx1] * (alpha1_new - alpha1_old) - y[idx2] * g[idx1][idx2] * (alpha2_new - alpha2_old)
            b2 = -ee2 - y[idx1] * g[idx1][idx2] * (alpha1_new - alpha1_old) - y[idx2] * g[idx2][idx2] * (alpha2_new - alpha2_old)
            self.b += (b1 + b2) * 0.5
            # print("Epoch:", i, ee1, ee2, eta, "Index 1:", idx1, alpha1_old, alpha1_new, "Index 2:", idx2, alpha2_old, alpha2_new)
            # print("Epoch:", i, b1, b2)
            print("Epoch:", i)
            print(self.alpha)
            print(self.b)

    def pol_kernel(self, x, y, p=1):
        return (x.dot(y.T) + 1) ** p

    def rbf_kernel(self, x, y, gamma=0.1):
        s = x[..., None, :]
        s = np.sum((x[..., None, :] - y) ** 2, axis=2)
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    def get_gram(self, x):
        return np.dot(x, x.T)

    def predict(self):
        y = np.asarray(self.t_data, np.float32)
        x = np.asarray(self.data, np.float32)
        k = self.pol_kernel(x, y)
        y_pred = np.dot(self.alpha.T, k) + self.b
        return y_pred

    def test(self):
        n = len(self.t_data)
        correct = 0
        y = self.predict()
        # print(y)
        # print(self.alpha)
        # print(self.b)
        for i in range(0, n):
            if y[i] * self.t_label[i] > 0:
                correct += 1
        print("Total:", n)
        print("Correct", correct)
        print("Accuracy:", float(correct) / float(n))

if __name__ == '__main__':
    gd = SVMHandler()
    gd.getData('bank-additional-full.csv')
    gd.train_gd()
    gd.test()
    perp = SVMHandler()
    perp.getData('bank-additional-full.csv')
    perp.train_perp()
    perp.test()
    # smo = SVMHandler()
    # smo.getData('bank-additional.csv')
    # smo.train_smo()
    # smo.test()

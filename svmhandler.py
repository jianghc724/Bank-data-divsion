from csvloader import CSVLoader
import numpy as np

class SVMHandler():

    def __init__(self):
        self.data = None
        self.label = None
        self.duration = None
        self.alpha = None
        self.b = None

    def getData(self, path):
        loader = CSVLoader()
        self.data, self.duration, self.label = loader.getData('bank-additional.csv')

    def train_gd(self, batch_size=128, epoch=1000, c=1, gamma=0.1):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        k = self.rbf_kernel(x, x, gamma)
        self.alpha = np.zeros(len(x))
        self.b = 0.
        for i in range(0, epoch):
            batch = np.random.choice(len(x), batch_size)
            x_batch = x[batch]
            y_batch = y[batch]
            e = - y_batch * (k.dot(self.alpha) + self.b)
            if np.max(e) <= 0:
                continue
            mask = err > 0
            delta = learning_rate * c *  y_batch[mask]
            self.alpha = self.alpha - 0.5 * learning_rate * (k.dot(self.alpha) + np.diag(k) * self.alpha) + np.sum(delta[..., None] * k[mask], axis=0)
            self.b += np.sum(delta)

    def train_smo(self, batch_size=128, epoch=1000, c=1, gamma=0.1):
        x = np.asarray(self.data, np.float32)
        y = np.asarray(self.label, np.float32)
        k = self.rbf_kernel(x, x, gamma)
        self.alpha = np.zeros(len(x))
        self.b = 0.
        for i in range(0, epoch):
            con1 = alpha > 0
            con2 = alpha < c
            y_pred = self.alpha.dot(k) + self.b
            err1 = y * y_pred - 1
            err2 = err1.copy()
            err3 = err3.copy()
            err1[(con1 & (err1 <= 0)) | (~con1 & (err1 > 0))] = 0
            err2[((~con1 | ~con2) & (err2 != 0)) | ((con1 & con2) & (err2 == 0))] = 0
            err3[(con2 & (err3 >= 0)) | (~con2 & (err3 < 0))] = 0
            err = err1 ** 2 + err2 **2 + err3 ** 2
            idx1 = np.argmax(err)
            idx2 = np.random.randint(len(y))
            while idx2 == idx1:
                idx2 = np.random.randint(len(y))
            eta = k[idx1][idx1] + k[idx2][idx2] - 2 * k[idx1][idx2]
            ee1 = y_pred[idx1] - y[idx1]
            ee2 = y_pred[idx2] - y[idx2]
            alpha1_old = self.alpha[idx1]
            alpha2_old = self.alpha[idx2]
            alpha1_new = alpha1_old - y[idx1] * (ee1 - ee2) / eta
            lower_bound = 0
            upper_bound = c
            if y[idx1] != y[idx2]:
                lower_bound = max(0., alpha1_old - alpha2_old)
                upper_bound = min(c, c + alpha1_old - alpha2_old)
            else:
                lower_bound = max(0., alpha1_old + alpha2_old - c)
                upper_bound = min(c, alpha1_old + alpha2_old)
            if alpha1_new > upper_bound:
                alpha1_new = upper_bound
            elif alpha1_new < lower_bound:
                alpha1_new = lower_bound
            alpha2_new = alpha2_old + y[idx1] * y[idx2] * (alpha1_old - alpha1_new)
            self.alpha[idx1] = alpha1_new
            self.alpha[idx2] = alpha2_new
            b1 = -ee1 - y[idx1] * k[idx1][idx1] * (alpha1_new - alpha1_old) - y[idx2] * k[idx1][idx2] * (alpha2_new - alpha2_old)
            b2 = -ee2 - y[idx1] * k[idx1][idx2] * (alpha1_new - alpha1_old) - y[idx2] * k[idx2][idx2] * (alpha2_new - alpha2_old)
            self.b += (b1 + b2) * 0.5



    def pol_kernel(self, x, y, p):
        return (x.dot(y.T) + 1) ** p

    def rbf_kernel(self, x, y, gamma=0.1):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    def predict(self, x):
        x = np.atleast_2d(x).astype(np.float32)
        k = self.rbf_kernel(self.data, x)
        y_pred = self.alpha.dot(k) + self.b
        return y_pred


if __name__ == '__main__':
    svmhandler = SVMHandler()
    svmhandler.getData('bank-additional.csv')

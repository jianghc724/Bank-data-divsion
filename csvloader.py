import csv
import random

job = ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"]
marital = ["divorced","married","single","unknown"]
education = ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]
bool = ["no","yes","unknown"]
con = ["cellular","telephone"]
month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
day = ["mon","tue","wed","thu","fri"]
poutcome = ["failure","nonexistent","success"]

class CSVLoader():

    def __init__(self):
        pass

    def getData(self, path, type=1):
        client = []
        contact = []
        social = []
        other = []
        label = []
        duration = []
        t_client = []
        t_contact = []
        t_social = []
        t_other = []
        t_label = []
        t_duration = []
        with open(path, newline='') as f:
            freader = csv.reader(f, delimiter=';')
            t = 0
            f = 0
            for row in list(freader)[1:]:
                row[0] = int(int(row[0]) / 10)
                for i in range(0, len(job)):
                    if row[1] == job[i]:
                        row[1] = i
                        break
                for i in range(0, len(marital)):
                    if row[2] == marital[i]:
                        row[2] = i
                        break
                for i in range(0, len(education)):
                    if row[3] == education[i]:
                        row[3] = i
                        break
                for i in range(0, len(bool)):
                    if row[4] == bool[i]:
                        row[4] = i
                        break
                for i in range(0, len(bool)):
                    if row[5] == bool[i]:
                        row[5] = i
                        break
                for i in range(0, len(bool)):
                    if row[6] == bool[i]:
                        row[6] = i
                        break
                for i in range(0, len(contact)):
                    if row[7] == con[i]:
                        row[7] = i
                        break
                for i in range(0, len(month)):
                    if row[8] == month[i]:
                        row[8] = i
                        break
                for i in range(0, len(day)):
                    if row[9] == day[i]:
                        row[9] = i
                        break
                row[10] = int(row[10])
                row[11] = int(row[11])
                row[12] = int(row[12])
                row[13] = int(row[13])
                for i in range(0, len(poutcome)):
                    if row[14] == poutcome[i]:
                        row[14] = i
                        break
                row[15] = float(row[15])
                row[16] = float(row[16])
                row[17] = float(row[17])
                row[18] = float(row[18])
                row[19] = float(row[19])
                if random.random() <= 0.7:
                    for i in range(0, len(bool)):
                        if row[20] == bool[i]:
                            label.append(i * 2 - 1)
                            client.append(row[0:7])
                            contact.append(row[7:10])
                            duration.append(row[10])
                            other.append(row[11:15])
                            social.append(row[15:20])
                            if i == 1:
                                t += 1
                            else:
                                f += 1
                            break
                else:
                    for i in range(0, len(bool)):
                        if row[20] == bool[i]:
                            t_client.append(row[0:7])
                            t_contact.append(row[7:10])
                            t_duration.append(row[10])
                            t_other.append(row[11:15])
                            t_social.append(row[15:20])
                            t_label.append(i * 2 - 1)
                            break
        print(t, f)
        while (t < f) and (type != 2):
            t1 = t
            for i in range(0, t1):
                if label[i] == 1:
                    label.append(label[i])
                    client.append(client[i])
                    contact.append(contact[i])
                    duration.append(duration[i])
                    other.append(other[i])
                    social.append(social[i])
                    t += 1
                    if t == f:
                        break
        print(t, f)
        return client, contact, duration, other, social, label, t_client, t_contact, t_duration, t_other, t_social, t_label

if __name__ == '__main__':
    csvloader = CSVLoader()
    data, contact, duration, other, social, label, t_client, t_contact, t_duration, t_other, t_social, t_label = csvloader.getData('bank-additional.csv')
    # print(data)
    # print(contact)
    # print(duration)
    # print(other)
    # print(social)
    # print(label)

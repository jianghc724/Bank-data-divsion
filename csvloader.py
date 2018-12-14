import csv

job = ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"]
marital = ["divorced","married","single","unknown"]
education = ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]
bool = ["no","yes","unknown"]
contact = ["cellular","telephone"]
month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
day = ["mon","tue","wed","thu","fri"]
poutcome = ["failure","nonexistent","success"]

class CSVLoader():

    def __init__(self):
        pass

    def getData(self, path):
        result = []
        label = []
        duration = []
        with open(path, newline='') as f:
            freader = csv.reader(f, delimiter=';')
            for row in list(freader)[1:]:
                row[0] = int(row[0])
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
                    if row[7] == contact[i]:
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
                duration.append(int(row[10]))
                row[10] = int(row[11])
                row[11] = int(row[12])
                row[12] = int(row[13])
                for i in range(0, len(poutcome)):
                    if row[14] == poutcome[i]:
                        row[13] = i
                        break
                row[14] = float(row[15])
                row[15] = float(row[16])
                row[16] = float(row[17])
                row[17] = float(row[18])
                row[18] = float(row[19])
                for i in range(0, len(bool)):
                    if row[20] == bool[i]:
                        label.append(i * 2 - 1)
                        break
                result.append(row[:-2])
        return result, duration, label

if __name__ == '__main__':
    csvloader = CSVLoader()
    data, duration, label = csvloader.getData('bank-additional.csv')
    print(data)
    print(duration)
    print(label)

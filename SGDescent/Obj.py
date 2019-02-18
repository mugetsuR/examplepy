import csv


class Objects:

    def __init__(self, path):
        self.path = path
        self.list = []

    def fromCsvToObjs(self):
        for row in csv.reader(open(self.path)):
            list.append(row)
from project.models.identify_dataset import DatasetIdentification
import os

class Sample(DatasetIdentification):
    def find_dataset(self):
        print('method implemented!')

def main():
    cwd = os.getcwd()
    print(cwd)
    t = Sample()
    t.find_dataset()
    t.generate_dataset_citation()

if __name__ == '__main__':
    main()
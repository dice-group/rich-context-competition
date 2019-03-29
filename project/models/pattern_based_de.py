import subprocess, os
from time import time
from project.models.identify_dataset import DatasetIdentification

class PatternDatasetExtraction(DatasetIdentification):
    """
    runs the pattern-based extractor using run.sh
    """
    def find_dataset(self):
        print('pattern-based dataset extraction begins..')

        start = time()
        os.chdir('project/models/pattern-based-dataset-extraction/')
        subprocess.Popen('pwd', shell=True)
        subprocess.call('./build.sh')
        subprocess.call('./run.sh')
        print(f'Pattern-based datasets identified in: {time() - start}')
        os.chdir('../../../')

def main():
    pde = PatternDatasetExtraction()
    pde.find_dataset()

if __name__ == '__main__':
    main()


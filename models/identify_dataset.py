from abc import ABCMeta, abstractmethod
import json


class DatasetIdentification(object):
    """
    Abstarct base class to identify datasets in a publication
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.dataset_citation_list = list()
        self.dataset_mention_list = list()

    def generate_dataset_citation(self, dataset_citation_list=None, output_path='data/output/'):
        """
        creates the json file data_set_citations.json
        :param dataset_citation_list:
        :return:
        """
        with open(output_path+"data_set_citations.json", "w") as write_file:
            json.dump(dataset_citation_list, write_file, indent=4)

    def generate_dataset_mention(self, dataset_mention_list=None, output_path='data/output/'):
        """
        creates the json file data_set_mentions.json
        :return:
        """
        with open(output_path+"data_set_mentions.json", "w") as write_file:
            json.dump(dataset_mention_list, write_file, indent=4)

    @abstractmethod
    def find_dataset(self):
        """
        pattern-based or RASA-based approach to find datasets
        returns a list of dictionaries according to the output format
        """
        raise NotImplementedError('The method must be defined first.')


def main():
    obj = DatasetIdentification()
    obj.generate_dataset_mention()

if __name__ == '__main__':
    main()
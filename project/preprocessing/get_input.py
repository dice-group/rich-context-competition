import pandas as pd
import os

data_path = "data/input/"
citation_path = "project/train_test/data_set_citations.json"
dataset_path = "project/train_test/data_sets.json"


class ProcessInput:
    def __init__(self, pub_df=None, dataset_df=None, citation_df=None):
        self.pub_df = pub_df
        self.dataset_df = dataset_df
        self.citation_df = citation_df

    def load_publication_input(self, path=data_path):
        """
        reads publications.json and converts it into a pandas dataframe.
        Also stores the text from the publications in a new column 'text'.
        pub_df.cols = 'pdf_file_name', 'pub_date', 'publication_id', 'text_file_name', 'title', 'unique_identifier', 'text'
        :param path:
        :return:
        """
        publications_path = os.path.abspath(os.path.join(str(path), 'publications.json'))
        with open(publications_path, 'r') as file:
            self.pub_df = pd.read_json(publications_path)

        text_file_path = os.path.join(str(path),"files/text/")
        pub_text_list = list()

        for _, row in self.pub_df.iterrows():
            with open(text_file_path + row['text_file_name'], 'r', errors='ignore') as txt_file:
                data = txt_file.read()
                pub_text_list.append(data)

        self.pub_df['text'] = pub_text_list
        return self.pub_df

    def load_dataset_vocab(self, path=dataset_path):
        with open(path) as f:
            self.dataset_df = pd.read_json(f)
        return self.dataset_df

    def load_citation_file(self, path=citation_path):
        with open(path) as f:
            self.citation_df = pd.read_json(f)
        return self.citation_df


def main():
    pi = ProcessInput()
    pub_df = pi.load_dataset_vocab()
    print(pub_df.columns.tolist())
    print(pub_df.shape)


if __name__ == '__main__':
    main()

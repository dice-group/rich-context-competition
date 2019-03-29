import pandas as pd
import os, re, json

data_path = "data/input/"
citation_path = "project/train_test/data_set_citations-train.json"
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
        with open(publications_path, 'r'):
            self.pub_df = pd.read_json(publications_path)

        text_file_path = os.path.join(str(path),"files/text/")
        pub_text_list = list()
        jel_method_list = list()
        jel_field_list = list()

        with open('project/train_test/jel-dict.json', 'r+') as file:
            jel_dict = json.load(file)

        for _, row in self.pub_df.iterrows():
            with open(text_file_path + row['text_file_name'], 'r', errors='ignore') as txt_file:
                data = txt_file.read()
                pub_text_list.append(data)

            jel_field = []
            jel_method = []

            jel_match = re.search(
                'JEL-?Classification:?\s?\n?|JEL\s?-?[c|C]lassification:?\s?\n?|JEL-?CLASSIFICATION:?\s?\n?|JEL\s?-?\s?[c|C][o|O][d|D][e|E]:?\s?\n?|JEL:?\s?\n?',
               data)
            if jel_match:
                selected_data = data[jel_match.end():]
                jel_line = selected_data[:selected_data.find('\n')]
                # print(f'file name', row['text_file_name'])
                # print(f'input_data: ',jel_line)

                matches = re.finditer(r"[a-zA-Z]\d+", jel_line, re.MULTILINE)
                for match in matches:
                    code = jel_line[match.start():match.end()][:3]
                    if code in jel_dict:
                        if code.startswith('c') or code.startswith('C'):
                            jel_method.append(jel_dict[code])
                        else:
                            jel_field.append(jel_dict[code])
            jel_field_list.append(jel_field)
            jel_method_list.append(jel_method)

        self.pub_df['text'] = pub_text_list
        self.pub_df['jel_method'] = jel_method_list
        self.pub_df['jel_field'] = jel_field_list
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

from rasa_nlu.model import Interpreter
from project.models.identify_dataset import DatasetIdentification
from operator import itemgetter
from preprocessing.get_input import ProcessInput
from project.preprocessing.preprocess_publications import PublicationPreprocessing
import os, json, logging
from time import time

class RasaDatasetExtraction(DatasetIdentification):


    def __init__(self, path='data/input/'):
        super().__init__()
        self.pi = ProcessInput()
        self.pubpr = PublicationPreprocessing()
        self.pub_df = self.pi.load_publication_input(path=path)
        self.dataset_vocab = self.pi.load_dataset_vocab()

    def extract_content(self, path):
        """
        loads sections of an article from the processed_articles/
        :param path:
        :return: a list containing sections
        """
        sections = []
        try:
            with open(path) as f:
                data = json.load(f)
                for key in data:
                    if "subject" == key:
                        continue
                    elif "keywords" == key:
                        sections.append(" ".join(data["keywords"]))
                    else:
                        sections.append(data[key].replace('\n', ' '))
        except:
            return sections

        return sections

    def find_entities(self, interpreter, threshold, file_name, text_content_path="project/additional_files/processed_articles/"):
        """
        finds entity mentions in different sections of an article
        :param interpreter: rasa model
        :param threshold: confidence score
        :param file_name:
        :param text_content_path:
        :return: a list containing a dict of distinct entities with their confidence score
        """
        entities = []
        sections = self.extract_content(os.path.join(text_content_path, file_name))

        for sent in sections:
            parsed = interpreter.parse(str(sent))
            for findings in parsed['entities']:
                start = findings['start']
                end = findings['end']
                confidence = findings['confidence']
                value = str(sent)[start:end]
                if confidence > threshold:
                    if value not in [ent['data_set'] for ent in entities]:
                        entities.append(
                            {'data_set': value, 'confidence': confidence})
                    else:
                        for ent in entities:
                            if ent['data_set'] == value and ent['confidence'] < confidence:
                                ent['confidence'] = confidence
                                break
        return entities

    def find_dataset(self, threshold=0.70):
        """
        runs the model for entity extraction
        :return: citation and mention lists
        """

        model_directory = "project/models/model_20181107-230703"

        all_interpreter = Interpreter.load(model_directory)

        dataset_citations = []
        dataset_mentions = []

        for _, row in self.pub_df.iterrows():
            try:
                print(row['pdf_file_name'])
                distinct_datasets = {} # { id1 : [('data1', score1), ('data2', score2)] , id2 : ..}
                entities = self.find_entities(all_interpreter, threshold, row['text_file_name'])

                for ent in entities:

                    if len(entities) == 1 and ent['data_set'].lower() == 'sage':
                        continue

                    for _, rowd in self.dataset_vocab.iterrows():
                        if ent['data_set'] in (mention for mention in rowd['mention_list']) or ent[
                        'data_set'] == rowd['name']:
                            if rowd['data_set_id'] not in distinct_datasets:
                                distinct_datasets[rowd['data_set_id']] = [(ent['data_set'], ent['confidence'])]
                            else:
                                distinct_datasets[rowd['data_set_id']].append((ent['data_set'], ent['confidence']))
                            break

                    result_dict = {}
                    result_dict["publication_id"] = row["publication_id"]
                    result_dict["score"] = ent['confidence']
                    result_dict["mention"] = ent['data_set']
                    dataset_mentions.append(result_dict)

                for id in distinct_datasets:
                    result_dict = {}
                    result_dict["publication_id"] = row["publication_id"]
                    result_dict["data_set_id"] = id
                    result_dict["score"] = max(distinct_datasets[id],key=itemgetter(1))[1]
                    result_dict["mention_list"] = [i[0] for i in distinct_datasets[id]]
                    dataset_citations.append(result_dict)

            except Exception as e:
                logging.exception(e)
                continue

        return dataset_citations, dataset_mentions


def main():
    cwd = os.getcwd()
    print(cwd)
    start_time = time()
    t = RasaDatasetExtraction()
    dataset_citations, dataset_mentions = t.find_dataset(threshold=0.70)
    t.generate_dataset_citation(dataset_citations)
    t.generate_dataset_mention(dataset_mentions)
    print(time() - start_time)


if __name__ == '__main__':
    main()

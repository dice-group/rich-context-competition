from rasa_nlu.model import Interpreter
from project.models.identify_dataset import DatasetIdentification
from operator import itemgetter
from project.preprocessing.get_input import ProcessInput
from project.preprocessing.preprocess_publications import PublicationPreprocessing
import spacy
import os, json
from time import time

class RasaDatasetExtraction(DatasetIdentification):


    def __init__(self, path='data/input/'):
        super().__init__()
        self.pi = ProcessInput()
        self.pubpr = PublicationPreprocessing()
        self.pub_df = self.pi.load_publication_input(path=path)
        self.dataset_vocab = self.pi.load_dataset_vocab()

    def extract_main_content(self, path = None):
        text = []
        try:
            with open(path) as f:
                data = json.load(f)
                for key in data:
                    if "abstract" == key:
                        text.append(data["abstract"])
                    elif "methodology" == key:
                        text.append(data["methodology"])
        except:
            return ''
        return '\n'.join(text)

    def extract_other_content(self, path = None):
        text = []
        try:
            with open(path) as f:
                data = json.load(f)
                for key in data:
                    if "subject" == key or "abstract" == key or "methodology" == key:
                        continue
                    elif "keywords" == key:
                        text.append(" ".join(data["keywords"]))
                    else:
                        text.append(data[key])
        except:
            return ''
        return '\n'.join(text)

    def find_entities(self, entities, interpreter, sentences):
        """
        finds entity mentions in sentences
        :param entities:
        :param interpreter:
        :param sentences:
        :return: a list containing a dict of distinct entities with their confidence score
        """
        # start = time()
        for sent in sentences:
            parsed = interpreter.parse(str(sent))
            for findings in parsed['entities']:
                start = findings['start']
                end = findings['end']
                #print(str(sent)[start:end])
                confidence = findings['confidence']
                #value = findings['value']
                value = str(sent)[start:end]
                if confidence > 0.65:
                    if value not in [ent['data_set'] for ent in entities]:
                        entities.append(
                            {'data_set': value, 'confidence': confidence})
                    else:
                        for ent in entities:
                            if ent['data_set'] == value and ent['confidence'] < confidence:
                                # print('entity present already: ',ent['data_set'], ' updated confidence: ', ent['confidence'])
                                ent['confidence'] = confidence
                                break
        # print(f'entities found in: {time() - start}')
        return entities

    def find_dataset(self):
        nlp = self.pubpr.nlp

        model_directory = "project/models/model_20181107-230703"
        main_model_dir = "project/models/model_20181128-182808"
        other_model_dir = "project/models/model_20181129-120423"
        text_content_path = "project/additional_files/processed_articles"

        # all_interpreter = Interpreter.load(model_directory)
        main_interpreter = Interpreter.load(main_model_dir)
        other_interpreter = Interpreter.load(other_model_dir)
        dataset_citations = []
        dataset_mentions = []

        # start = time()
        for _, row in self.pub_df.iterrows():
            try:
                print(row['pdf_file_name'])
                entities = []
                distinct_datasets = {} # { id1 : [('data1', score1), ('data2', score2)] , id2 : ..}
                main_data = self.extract_main_content(os.path.join(text_content_path, row['text_file_name'])).replace('\n', ' ')
                other_data = self.extract_other_content(os.path.join(text_content_path, row['text_file_name'])).replace('\n', ' ')
                main_doc = nlp(main_data) # only abstract & methodology/data
                other_doc = nlp(other_data) # other content
                main_sentences = main_doc.sents
                other_sentences = other_doc.sents
                # print(f'main sentences: {main_data}')

                self.find_entities(entities, main_interpreter, main_sentences)
                self.find_entities(entities, other_interpreter, other_sentences)

                #print(entities)

                for ent in entities:
                    #found_match = False
                    found = 0
                    for _, rowd in self.dataset_vocab.iterrows():
                        if ent['data_set'] in (mention for mention in rowd['mention_list']) or ent[
                            'data_set'] == rowd['name']:
                        # if ent['data_set'].lower() in (mention.lower() for mention in rowd['mention_list']) or ent['data_set'].lower() == rowd['name'].lower():
                            if rowd['data_set_id'] not in distinct_datasets:
                                distinct_datasets[rowd['data_set_id']] = [(ent['data_set'], ent['confidence'])]
                            else:
                                distinct_datasets[rowd['data_set_id']].append((ent['data_set'], ent['confidence']))
                            # found += 1
                            # if found == 3:
                            break

                    #if (not found_match):
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
                    # for i in distinct_datasets[id]:
                    #     print('mention list contains:', i[0], i)
                    # print('mention list: ', result_dict["mention_list"])
                    dataset_citations.append(result_dict)

                # print(f'time taken per file: {time() - start}')

            except Exception as e:
                print(e)
                continue

        print(dataset_citations)
        print(dataset_mentions)

        return dataset_citations, dataset_mentions


def main():
    cwd = os.getcwd()
    print(cwd)
    start_time = time()
    t = RasaDatasetExtraction()
    dataset_citations, dataset_mentions = t.find_dataset()
    end_time = time()
    print(end_time - start_time)


if __name__ == '__main__':
    main()

from rasa_nlu.model import Interpreter
from project.models.identify_dataset import DatasetIdentification
from operator import itemgetter
from preprocessing.get_input import ProcessInput
from project.preprocessing.preprocess_publications import PublicationPreprocessing
import os, json, logging, re, spacy, string
from time import time
from spacy.lang.en.stop_words import STOP_WORDS
from statistics import median
import matplotlib.pyplot as plt

class RasaDatasetExtraction(DatasetIdentification):


    def __init__(self, path='data/input/'):
        super().__init__()
        self.pi = ProcessInput()
        self.pubpr = PublicationPreprocessing()
        self.pub_df = self.pi.load_publication_input(path=path)
        self.dataset_vocab = self.pi.load_dataset_vocab()
        self.nlp = spacy.load('en_vectors_web_lg', disable = ['tagger', 'parser', 'ner', 'textcat'])

    def extract_content(self, path):
        """
        loads sections of an article from the processed_articles/
        :param path:
        :return: a list containing sections
        """
        sections = {}
        try:
            with open(path) as f:
                data = json.load(f)
                for key in data:
                    if "subject" == key:
                        continue
                    elif "keywords" == key or "jel_field" == key or "jel_method" == key:
                        sections[key] = " ".join(data[key])
                    else:
                        sections[key] = data[key].replace('\n', ' ')
        except:
            return sections

        return sections

    def find_entities_jel_field(self, interpreter, threshold, file_name, text_content_path="project/additional_files/processed_articles/"):
        """
        finds entity mentions in different sections of an article
        :param interpreter: rasa model
        :param threshold: confidence score
        :param file_name:
        :param text_content_path:
        :return: a list containing a dict of distinct entities with their confidence score, fields as per jel-classification
        """

        entities = []
        jel_field = []
        sections = self.extract_content(os.path.join(text_content_path, file_name))

        for sect in sections:
            if sect == "jel_field":
                jel_field = str(sections[sect])

            parsed = interpreter.parse(str(sections[sect]))

            for findings in parsed['entities']:
                start = findings['start']
                end = findings['end']
                confidence = findings['confidence']
                value = str(sections[sect])[start:end]
                if confidence > threshold:
                    if value not in [ent['data_set'] for ent in entities]:
                        entities.append(
                            {'data_set': value, 'confidence': confidence})
                    else:
                        for ent in entities:
                            if ent['data_set'] == value and ent['confidence'] < confidence:
                                ent['confidence'] = confidence
                                break
        return entities, jel_field

    def dataset_belongsTo_field(self, research_field, jel_field, dataset_subject, dataset_description):
        """
        checks if the dataset predicted by rasa lies in the domain of research field
        :param research_field: predicted by the word2vec model
        :param dataset_subject: from train_test/data_sets.json
        :return:
        """
        start = time()

        if len(jel_field) == 0 and len(re.split(",|;", research_field)) == 1: #field predicted by model only
            return True

        all_fields = (jel_field + " ," + research_field).strip()
        all_fields = re.sub(r'\d+', '', all_fields) #remove numbers
        wordsInFields = [word for word in all_fields.lower().split() if word not in STOP_WORDS]

        for field in re.split(",|;", ' '.join(wordsInFields)):
            if len(field) < 2:
                continue

            rfield = self.nlp(field)

            if len(dataset_subject) == 0 and len(dataset_description) > 0:
                wordsInDesc = [word for word in dataset_description.lower().split() if word not in STOP_WORDS]
                desc = self.nlp(' '.join(wordsInDesc))
                if rfield.similarity(desc) > 0.68:
                    return True

            elif len(dataset_subject) > 0:
                for term in dataset_subject.split(','):
                    term = re.sub(r'\d+', '', term)
                    term = re.sub('[' + string.punctuation + ']', '', term)
                    if (len(term)) < 2:
                        continue
                    subject_term = self.nlp(term.strip())
                    if rfield.similarity(subject_term) > 0.68:
                        return True

            elif len(dataset_subject) == 0 and len(dataset_description) == 0:
                return True

        return False

    def find_dataset(self, threshold=0.70):
        """
        runs the model for entity extraction
        :return: citation and mention lists
        """

        model_directory = "project/models/model_20190116-214229"

        all_interpreter = Interpreter.load(model_directory)

        dataset_citations = []
        dataset_mentions = []
        count_jel_methods = 0

        for _, row in self.pub_df.iterrows():
            if len(row['jel_method']) > 0:
                count_jel_methods += 1

            try:
                print(row['pdf_file_name'])
                distinct_datasets = {} # { id1 : [('data1', score1), ('data2', score2)] , id2 : ..}
                entities, jel_field = self.find_entities_jel_field(all_interpreter, threshold, row['text_file_name'])

                with open('project/additional_files/pub_field.json', 'r+') as file:
                    pub_field_dict = json.load(file)

                for ent in entities:

                    belongsToField = True

                    if len(entities) == 1 and ent['data_set'].lower() == 'sage':
                        continue

                    for _, rowd in self.dataset_vocab.iterrows():

                        if ent['data_set'] in (mention for mention in rowd['mention_list']) or ent[
                        'data_set'] == rowd['name']:

                            if not self.dataset_belongsTo_field(pub_field_dict[row['pdf_file_name'].replace('.pdf','').strip()], jel_field, rowd["subjects"], rowd["description"]):
                                belongsToField = False
                                break

                            if rowd['data_set_id'] not in distinct_datasets:
                                distinct_datasets[rowd['data_set_id']] = [(ent['data_set'], ent['confidence'])]
                            else:
                                distinct_datasets[rowd['data_set_id']].append((ent['data_set'], ent['confidence']))
                            break

                    if not belongsToField:
                        print(f"not field: {ent['data_set']}")
                        continue

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

        # write results from rasa to intermediate files
        with open("project/additional_files/data_set_citations_rasa.json", "w+") as write_file:
            json.dump(dataset_citations, write_file, indent=4, ensure_ascii=False)

        with open("project/additional_files/data_set_mentions_rasa.json", "w+") as write_file:
            json.dump(dataset_mentions, write_file, indent=4, ensure_ascii=False)

        return dataset_citations, dataset_mentions


def main():
    cwd = os.getcwd()
    print(cwd)
    t = RasaDatasetExtraction()
    dataset_citations, dataset_mentions = t.find_dataset(threshold=0.72)
    t.generate_dataset_citation(dataset_citations)
    t.generate_dataset_mention(dataset_mentions)


if __name__ == '__main__':
    main()

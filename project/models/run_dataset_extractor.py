import json
from project.models.rasa_based_de import RasaDatasetExtraction
from project.models.pattern_based_de import PatternDatasetExtraction
from project.preprocessing.get_input import ProcessInput
import matplotlib.pyplot as plt
from statistics import median
import matplotlib.ticker as mticker
import numpy as np
from operator import itemgetter
from itertools import groupby
from time import time
from collections import OrderedDict
import logging

class DatasetCombiner:

    def __init__(self,path='data/input/'):
        self.rde = RasaDatasetExtraction()
        self.pde = PatternDatasetExtraction()
        self.pi = ProcessInput()
        self.dataset_vocab = self.pi.load_dataset_vocab()
        self.pub_df = self.pi.load_publication_input(path=path)

    def combine_de_approaches(self, threshold = 0.72):
        start1 = time()

        # run pattern-based dataset-extractor
        self.pde.find_dataset()

        # load result and find dataset-mention frequecy
        dataset_freq, pattern_based_results = self.find_patternBased_dataset_freq(plot=False)

        # --- run rasa-based dataset-extractor ---
        self.rde.find_dataset(threshold=threshold)

        with open("project/additional_files/data_set_citations_rasa.json", "r+") as file:
            rasa_datasets = json.load(file)

        # --- update the publication dataframe by inserting new columns that will hold the output from both the approaches --- #
        self.pub_df['pattern_dataset_citations'] = np.nan
        self.pub_df['pattern_dataset_citations'] = self.pub_df['pattern_dataset_citations'].astype(object)

        self.pub_df['rasa_dataset_citations'] = np.nan
        self.pub_df['rasa_dataset_citations'] = self.pub_df['rasa_dataset_citations'].astype(object)

        with open("project/additional_files/data_set_mentions_rasa.json", "r+") as file:
            rasa_mentions = json.load(file)

        irrelevant_mentions = {}

        # --- add datasets from both the approaches to the dataframe ---
        for index, row in self.pub_df.iterrows():

            pattern_dataset_citations = []
            rasa_dataset_citations = []

            # --- remove irrelevant dataset citations from Simple Dataset Mention Search's (pattern-based) output ---
            self.remove_irrelevant_pattern_based_mentions(dataset_freq, pattern_based_results,
                                                          pattern_dataset_citations, row)

            self.pub_df.at[index, 'pattern_dataset_citations'] = pattern_dataset_citations

            # --- filter irrelevant dataset citations from rasa output ---
            self.remove_irrelevant_rasa_mentions(dataset_freq, irrelevant_mentions, rasa_dataset_citations,
                                                 rasa_datasets, row)
            self.pub_df.at[index, 'rasa_dataset_citations'] = rasa_dataset_citations

        # --- remove irrelevant citations from rasa output ---
        rasa_mentions = self.remove_irrelevant_rasa_citations(irrelevant_mentions, rasa_mentions)

        dataset_citations = []
        dataset_mentions = []


        # --- take a union of dataset citations from the two approaches ---
        start = time()
        for _, row in self.pub_df.iterrows():
            try:
                all_citations = list(row['rasa_dataset_citations'])+ \
                               list(row['pattern_dataset_citations']) # concatenation
                sorted_citations = sorted(all_citations, key=itemgetter('data_set_id'))  # sort citations on data_set_id
                distinct_citations = []
                i=0
                visited = [False] * len(sorted_citations)

                # --- if the next citation obj has the same dataset_id as previous one, take a union of the mention_list ---
                while(i+1 < len(sorted_citations)):
                    pub_id = sorted_citations[i]['publication_id']
                    dataset_id = sorted_citations[i]['data_set_id']
                    if sorted_citations[i]['data_set_id'] == sorted_citations[i+1]['data_set_id']:
                        all_mentions = list(set(sorted_citations[i]['mention_list'] +
                                                sorted_citations[i+1]['mention_list']))
                        score = (sorted_citations[i]['score'] + sorted_citations[i+1]['score']) / 2

                        visited[i] = True
                        visited[i+1] = True
                        i += 2
                    else:
                        score = sorted_citations[i]['score']
                        all_mentions = sorted_citations[i]['mention_list']
                        visited[i] = True
                        i += 1

                    distinct_citations.append({'publication_id': pub_id,
                                               'data_set_id': dataset_id,
                                               'score': score, 'mention_list': all_mentions})
                    self.add_mention_to_dataset_mentions(all_mentions, dataset_mentions, pub_id, score)

                if i < len(visited) and visited[i] == False:
                    distinct_citations.append({'publication_id': sorted_citations[i]['publication_id'],
                                               'data_set_id': sorted_citations[i]['data_set_id'],
                                               'score': sorted_citations[i]['score'],
                                               'mention_list': sorted_citations[i]['mention_list']})
                    all_mentions = sorted_citations[i]['mention_list']
                    pub_id = sorted_citations[i]['publication_id']
                    score = sorted_citations[i]['score']
                    self.add_mention_to_dataset_mentions(all_mentions, dataset_mentions, pub_id, score)
                    visited[i] = True

                dataset_citations.extend(distinct_citations)
            except Exception as e:
                logging.exception(e)
                continue


        # --- remove common mentions ---
        dataset_mentions = self.remove_common_mentions(dataset_mentions, rasa_mentions)

        print(f"time to union: {time() - start}")
        # --- write to output files ---
        self.pde.generate_dataset_citation(dataset_citations)
        self.pde.generate_dataset_mention(rasa_mentions+dataset_mentions)
        print(f"time taken by entire combiner pipeline: {time() - start1}")

    def add_mention_to_dataset_mentions(self, all_mentions, dataset_mentions, pub_id, score):
        for mention in all_mentions:
            m_dict = {}
            m_dict['publication_id'] = pub_id
            m_dict['score'] = score
            m_dict['mention'] = mention
            dataset_mentions.append(m_dict)

    def remove_common_mentions(self, dataset_mentions, rasa_mentions):
        remove_mentions = []
        for mention in dataset_mentions:
            for rasa_mention in rasa_mentions:
                if mention['publication_id'] == rasa_mention['publication_id'] and \
                        mention['mention'] == rasa_mention['mention']:
                    remove_mentions.append(mention)
        dataset_mentions = [mention for mention in dataset_mentions if mention not in remove_mentions]
        return dataset_mentions

    def remove_irrelevant_rasa_citations(self, irrelevant_mentions, rasa_mentions):
        remove_mentions = []
        start2 = time()
        for citation in irrelevant_mentions:
            for mention in rasa_mentions:
                if citation == mention['publication_id'] and mention['mention'] in irrelevant_mentions[citation]:
                    remove_mentions.append(mention)
                    print(f'removed {mention}')
        rasa_mentions = [mention for mention in rasa_mentions if mention not in remove_mentions]
        print(f"time taken to remove fake datasets: {time() - start2}")
        return rasa_mentions

    def remove_irrelevant_rasa_mentions(self, dataset_freq, irrelevant_mentions, rasa_dataset_citations, rasa_datasets,
                                        row):
        for citation in rasa_datasets:
            try:
                if citation['publication_id'] == row['publication_id']:
                    if citation["data_set_id"] in dataset_freq and dataset_freq[citation["data_set_id"]] > 1.20 \
                            * median(set(dataset_freq.values())) and len(citation["mention_list"]) < 2:
                        if citation["publication_id"] not in irrelevant_mentions:
                            irrelevant_mentions[citation["publication_id"]] = citation["mention_list"]
                        else:
                            irrelevant_mentions[citation["publication_id"]].extend(citation["mention_list"])
                        continue
                    else:
                        rasa_dataset_citations.append(citation)
            except Exception as e:
                logging.exception(e)
                continue

    def remove_irrelevant_pattern_based_mentions(self, dataset_freq, pattern_based_results, pattern_dataset_citations,
                                                 row):
        for res in pattern_based_results:
            try:
                if res['publication_id'] == row["publication_id"]:
                    pattern_dataset_citations.append(res)
                    if dataset_freq[res["data_set_id"]] > 1.20 * median(set(dataset_freq.values())) and \
                            len(res["mention_list"]) < 3:  #if few instances in mention list, ignore
                        continue
                    elif len(res["mention_list"]) < 2:
                        continue
                    else:
                        pattern_dataset_citations.append(res)
            except Exception as e:
                logging.exception(e)
                continue

    def find_patternBased_dataset_freq(self, plot=True):
        with open('project/models/pattern-based-dataset-extraction/data_set_citations.json', 'r+') as file:
            results = json.load(file)
            dataset_freq = OrderedDict()

            for res in results:
                if res["data_set_id"] not in dataset_freq:
                    print(res)
                    dataset_freq[res["data_set_id"]] = 1
                else:
                    dataset_freq[res["data_set_id"]] += 1

        # --- plot the frequency distribution graph ---- #
        if plot:
            self.plot_freq_distribution(dataset_freq)

        return dataset_freq, results

    def plot_freq_distribution(self, dataset_freq):
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
        params = {'text.usetex': True,
                  'font.size': 17.5,
                  'font.family': 'lmodern',
                  'text.latex.unicode': True,
                  }
        plt.rcParams.update(params)
        plt.figure(figsize=(8, 8))
        plt.bar(range(12), list(dataset_freq.values())[:12], align='center')
        plt.xticks(range(12), sorted(list(dataset_freq.keys())[:12]))
        plt.xlabel('Dataset-IDs')
        plt.ylabel('Number of times a dataset was found')
        # plt.title('Frequency of the predicted datasets', fontsize=21, pad=20, fontname="Times New Roman Bold")
        # plt.tick_params(axis='both', which='major', pad=9, labelsize=6)
        plt.savefig('freq.pdf', dpi=1200, bbox_inches='tight', )
        plt.show()


def main():
    dc = DatasetCombiner()
    dc.combine_de_approaches()

if __name__ == '__main__':
    main()





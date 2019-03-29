import subprocess, os, json
from time import time
import logging
from operator import itemgetter
from matplotlib import pyplot as plt


class FieldMedthodIdentification:
    """
    Runs the jar file containing trained model from inside the shell script to identify fields and methods used.
    """
    def run_pipeline(self, output_path='data/output/'):
        print('field method detection begins..')

        start = time()
        os.chdir('project/models/word2vec/')
        subprocess.Popen('pwd', shell=True)
        subprocess.call('./run_w2v.sh')
        print(f'Fields and methods identified in: {time() - start}')
        os.chdir('../../../')

        research_fields_list = []
        methods_list = []
        pub_field_dict = {}

        blacklisted_fields = ['Case Study & Narrative Analysis', 'Evaluation', 'Evaluation Theory',
                              'Focus Group Research', "Performance Studies",
                              'Grounded Theory', 'Content Analysis', 'Meta - Analysis', 'Mixed Methods',
                              'Narrative Analysis',
                              'Qualitative Data Analysis', 'Qualitative Evaluation', 'Qualitative Research',
                              'Qualitative Software',
                              'Quantitative Evaluation', 'Quantitative / Statistical Research',
                              'Regression & Correlation', "Quantitative/Statistical Research",
                              'Research Design', 'Research Design Software', 'Research Ethics', 'Research Methods',
                              'Research Projects', 'Structural Equation Modeling', 'Statistical Computing Environments',
                              'Survey Research', 'Test & Measurement', 'Visual Methods', 'Digital Methods',
                              'Comparative Historical Analysis', 'Research Methods & Evaluation'
                              'Formal Modeling, Game Theory & Decision Theory', 'Using Software in Data Analysis',
                              'True Experimental Design in Psychology', 'Introductory Statistics for Psychology',
                              'Intermediate Statistics for Psychology', 'Advanced Statistics for Psychology',
                              "Marx, Durkheim & Weber", "ANOVA/MANOVA", "Formative Assessment", "Grading","Scandinavian Studies"]

        field_freq = {}
        # --- result generation for fields ---
        try:
            with open('project/additional_files/research_fields_results.json', 'r+') as file:
                results = json.load(file)
                for res in results:
                    res_dict = {}
                    key_terms = ''
                    jel_field = ''
                    with open("project/additional_files/processed_articles/" + res['fileName']) as f:
                        data = json.load(f)
                        for key in data:
                            if "keywords" == key:
                                keywords = ", ".join(data["keywords"]).strip()
                                print(keywords)
                                if (len(keywords) > 1):
                                    key_terms = keywords + ", "

                            elif "jel_field" == key:
                                jel_field = ", ".join(data["jel_field"]).strip()

                    res_dict['publication_id'] = int(res['fileName'].replace('.txt','').strip())
                    sorted_res = sorted(res['result'], key=itemgetter('score'), reverse=True)

                    i = 0
                    while sorted_res[i]['fieldLabel'] in blacklisted_fields:
                        i+=1
                        if i == len(sorted_res):
                            break

                    # -- if all the terms in top-10 results are blacklisted, use jel-field (if present) as research
                    # field, else mark the top-most term as field
                    if i == len(sorted_res):
                        if len(jel_field) > 1:
                            res_dict['research_field'] = jel_field
                            res_dict['score'] = 0.92
                        else:
                            res_dict['research_field'] = sorted_res[0]['fieldLabel']
                            res_dict['score'] = sorted_res[0]['score']
                    else:
                        res_dict['research_field'] = sorted_res[i]['fieldLabel']
                        res_dict['score'] = sorted_res[i]['score']

                    research_fields_list.append(res_dict)
                    pub_field_dict[res_dict['publication_id']] = (key_terms +res_dict['research_field']).strip()

                    # ---- to plot frequency distribution of predicted fields ---
                    if res_dict['research_field'] not in field_freq:
                        field_freq[res_dict["research_field"]] = 1
                    else:
                        field_freq[res_dict["research_field"]] += 1

        except Exception as e:
            logging.exception(e)

        # --- result generation for methods ---
        try:
            with open('project/additional_files/research_methods_results.json', 'r+') as file:
                results = json.load(file)
                for res in results:
                    sorted_res = sorted(res['result'], key=itemgetter('score'), reverse= True)
                    res_dict = {}
                    res_dict['publication_id'] = int(res['fileName'].replace('.txt','').strip())
                    res_dict['method'] = sorted_res[0]['methodLabel']
                    res_dict['score'] = sorted_res[0]['score']
                    methods_list.append(res_dict)
        except Exception as e:
            logging.exception(e)

        with open(output_path+"research_fields.json", "w+") as write_file:
            json.dump(research_fields_list, write_file, default=write_file, indent=4, ensure_ascii=False)

        with open(output_path+"methods.json", "w+") as write_file:
            json.dump(methods_list, write_file, default=write_file, indent=4, ensure_ascii=False)

        with open('project/additional_files/pub_field.json', 'w+') as write_file:
            json.dump(pub_field_dict,write_file, indent=4, ensure_ascii=False)

        with open('project/additional_files/field_freq.json', 'w+') as write_file:
            json.dump(field_freq, write_file, indent=4, ensure_ascii=False)

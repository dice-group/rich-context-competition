import subprocess, os, json
from time import time
import logging


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

        try:
            with open('project/additional_files/research_fields_results.json') as file:
                results = json.load(file)
                for res in results:
                    res_dict = {}
                    res_dict['publication_id'] = int(res['fileName'].replace('.txt','').strip())
                    res_dict['research_field'] = res['result'][0]['fieldLabel']
                    res_dict['score'] = res['result'][0]['score']
                    research_fields_list.append(res_dict)
        except Exception as e:
            logging.exception(e)

        try:
            with open('project/additional_files/research_methods_results.json') as file:
                results = json.load(file)
                for res in results:
                    res_dict = {}
                    res_dict['publication_id'] = int(res['fileName'].replace('.txt','').strip())
                    res_dict['method'] = res['result'][0]['methodLabel']
                    res_dict['score'] = res['result'][0]['score']
                    methods_list.append(res_dict)
        except Exception as e:
            logging.exception(e)

        with open(output_path+"research_fields.json", "w") as write_file:
            json.dump(research_fields_list, write_file, default=write_file, indent=4)

        with open(output_path+"methods.json", "w") as write_file:
            json.dump(methods_list, write_file, default=write_file, indent=4)

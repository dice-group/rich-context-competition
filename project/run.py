import sys
sys.path.append('/')

#from project.models.rasa_based_de import RasaDatasetExtraction
from project.models.run_dataset_extractor import DatasetCombiner
from project.models.identify_fields_methods import FieldMedthodIdentification
from project.preprocessing.preprocess_publications import PublicationPreprocessing
from time import time

start_time1 = time()
print(f'preprocessing begins..')
pp = PublicationPreprocessing()
pp.process_text(extract_np=True, write_processed_files=True) # necessary files created in project/additional_files for all pubs
print(f"preprocessing done in: {time() - start_time1}")

# identify research fields and methods

start_time2 = time()
fmi = FieldMedthodIdentification()
fmi.run_pipeline(output_path='data/output/')
print(f"Found methods and fields in: {time() - start_time2}")

# identify datasets and prepare output files - dataset_citations, dataset_mentions

start_time3 = time()
dc = DatasetCombiner(path='data/input/')
dc.combine_de_approaches(threshold=0.72)
print(f"Found datasets in: {time() - start_time3} ") # time_taken in seconds

print(f"entire pipeline took: {time() - start_time1} ")

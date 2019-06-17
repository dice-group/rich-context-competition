# Rich Cntext Competition

This repo contains our code for the [Rich Context Competition](https://coleridgeinitiative.org/richcontextcompetition). 

##### Run the entire project's pipleine by executing ```project/run.py```
<!-- ##### To read the documentation, click [here](https://docs.google.com/document/d/1-bUtlLfTK4x7-syyAQ2rBHq6C11DJeFXdI80Ii-BUCk/edit#) -->

### Table of Contents
**1. [Preprocessing](#preprocessing)**	
**2. [Dataset detection](#dataset-detection)**	
**3. [Identification of Research methods and fields](#identification-of-research-methods-and-fields)**	

### Executive Summary 
A high-level summary of our approach is given below:
1. Preprocessing <br>
2. Dataset identification using a combination of simple dataset-search and CRFs based on rasa-nlu <br>
3. Identification of methods and fields using vectorization and cosine similarity

<!-- To run the entire pipeline, execute project/run.py. --> 

### Preprocessing 

**To run the preprocessing module only, execute** `project/preprocessing/preprocess_publications.py`. Read on for a comprehensive overview of the preprocessing module. 

###### Generating text files from PDF:
While inspecting the input data, we found out that there were many text articles that weren’t converted properly from PDF and hence, contained mostly non-ASCII characters. In order to read such articles, we relied on the open source tool pdf2text from [poppler suite](https://manpages.debian.org/testing/poppler-utils/pdfinfo.1.en.html) to extract text files from PDFs. <br>

###### Before training and evaluating the models, we pre-process the given text articles to: 
1. Handle the words that get split by hyphens, 
2. Remove noisy data (equations, tables, acknowledgment, references, etc.), 
3. Identify and extract the main sections (such as abstract, keywords, methodology/data, summary etc.) and,
4. Extract noun phrases from these sections. 
5. If a section is not found in the article (because of no explicit mention), then only the sections that can be identified are extracted and the remaining content is saved as ‘reduced_content’ after cleaning.
6. We use the open source tool pdfinfo from the [poppler suite](https://manpages.debian.org/testing/poppler-utils/pdfinfo.1.en.html) to extract PDF metadata that seldom contains the keywords and subject of a research article.  <br>
<br>
After the preprocessing module is run, three folders are generated in `project/additional_files` (`pdf-info`, `nounPhrases` and `processed_articles`) which are used for training and evaluating the models. <br>

***************************************************

### Dataset detection 

**To run the entire dataset detection pipeline, run** `project/models/run_dataset_extractor.py`.  

The pipeline has been described below. Note, each component of the pipeline can be run seprately. <br>
For identifying the dataset mentions in a research article, we use two approaches.

**Simple dataset mention search:**
We choose the dataset mentions from the given `data_set.json` file which occur for one dataset only and used these unique mentions to search for the corresponding datasets in the text documents. 
Run `project/models/pattern_based_de.py` to generate the interim result file  `(project/models/pattern-based-dataset-extraction/data_set_citations.json)`. 

**Rasa-based Dataset Detection:**
We train an entity extraction model based on conditional random fields using Rasa NLU. The 7500 labeled publications from phase 1 corpus  (after preprocessing) are used for generating the training data for Rasa NLU.  
To run the trained-CRF model, jump directly to [running the trained-CRF model](#trained-CRF). <br>
Otherwise, specifications for model training are given below. <br>
 
**Training of the CRF-Model:**

**1. Generation of the training file for Rasa NLU:**

Configuration in the file `rasa/conf/conf.cnf`: <br>
```
data_set_citations = JSON File with the citations of the dataset 
publication_text_folder = Folder with publications 
trainingfile = Outputfile for the next step in JSON Format 
```
Run the python file `rasa/loadtrainingdata.py`

**2. Training of the CRF model:**

Configuration in the file `rasa/conf/conf.cnf`: <br>
```
rasa_conf = Configuration for the training process of rasa NLU (Details: https://rasa.com/docs/nlu/0.12.1/config/)
modelfolder = output folder for the model
trainingfile = generated File from step 1
```
Run the file `rasa/training.py`; this will generate the CRF model.

**3. Running the trained CRF-Model:**  <a id="trained-CRF"></a>
<br>
The trained models are run on preprocessed data from `project/additional_files/processed_articles/`. 

Run `project/models/rasa_based_de.py` to identify datasets and generate the interim output files, `data_set_citations_rasa.json`, and `data_set_mentions_rasa.json` in `project/additional_files/`. Only the entities that have a confidence score greater than a threshold value and belong to the research field of the article are considered as datasets and written to files. For checking if a dataset belongs to the field of research, we find the cosine similarity of the terms in `‘subjects’`  field of the dataset metadata (`data_sets.json`) with the keywords and identified research field of the article. 

**4. Combining the two approaches:** <br>
The output generated by the two approaches above is checked for fake mentions before being written to the final output file `data_set_citations.json`, and `data_set_mentions.json` in `data/output`.  This is done by calculating the frequency of dataset mentions and removing mentions that occur more than a threshold_value * median of dataset_frequency.

Run `project/models/run_dataset_extractor.py` to get the combined output from both approaches. 

************************************
### Identification of Research methods and fields
Run `project/models/identify_fields_methods.py` to run the complete methods and fields identification pipleine and generate output files. For executing the pipeline step-by-step, see below:

   **1. Preprocessing** <br>
  - **Word2Vec Model generation:** In this pre-processing step, we use the sample vocabulary files of research fields and methods (`sage_research_fields.csv` & `dbpedia_methods_np_vocab.json`) to generate a vector model for each research field and method in the vocabulary. The vector model is generated by using the labels and description of the available research fields and methods and then using the noun phrases present in them to form a sum vector. The sum vector is basically the sum of all the vectors of the words present in a particular noun phrase. `GoogleNews-vectors-negative300.bin` Word2vec model is used to extract the vectors of the individual words. <br>
  Following Java classes are called to generate the models:
    ```
    upb.dice.rcc.tool.rfld.generator.RsrchFldMdlGnrtrCsv
    upb.dice.rcc.tool.rmthd.generator.DbpFrmtRsrchMthdMdlGnrtr
    ```
    
  We get the following as an output of this preprocessing step: <br>
    - Word2Vec Research field model (`ResearchFields_NormalizedModel.bin`) <br>
    - Word2Vec Research method model (`StatisticalMethods_NormalizedModel.bin`)

   - **Research Method training results creation:** In this step, we generate a research methods result file for the publications present in the training set. The results are generated using a naïve “finder” algorithm which for each publication, selects the research method that has the highest cosine similarity to any of its noun phrase’s vector. This result is later used to assign weights to Research Methods using inverse document frequency.
We get the `research_methods_results_db.json` as an output to this step.

   **2. Processing**

   - **Finding Research Fields and Methods:** To find the research fields and methods for a given list of publications we perform the following steps (Steps “2”and “3” are executed iteratively for each publication): <br>
    1) **Naïve Research Method Finder run:** In this step, we execute a naïve research method finding algorithm for all the current publications and then merge the results with the existing result from the training. The combined result is then used to generate IDF weight values for each Research Method. <br>
    2) **Top Cosine Similarity Research Field Finder run:** In this step, we first find the closest research field from each noun phrase in the publication. Then we select the Top N(10) pairs that have the highest cosine similarity. Afterwards, the noun phrases with cosine similarity values less than a given threshold(0.9) are filtered out. The end result is then passed on to the post-processing algorithm. <br>
    3) **IDF based Research Method Finder:** Like the step before, we find the closest research method to each noun phrase and then sort the pairs based on their weighted cosine similarity. The weights used are from the IDF values generated in the first step and the manual weights assigned based on the section of publication where the noun phrase was extracted from. The pair with the highest weighted cosine similarity is then chosen.
	
   **The main method in** `upb.dice.rcc.multscr.main.RccMainMultScrIdfAdv` **Java class is called to execute the above-listed steps. All the code for training the model can be found [here](https://github.com/nikit91/Jword2vec/tree/rich-context)**. <br>

   **Total time taken to train the model - 11 mins.** 

   **3. Running the model against publications** <br>
    The model is run as a jar file `project/models/word2vec/RccWord2Vec.jar` from inside the shell script, `project/models/word2vec/run_w2v.sh`, which is executed in the python file, `project/models/identify_fields_methods.py`. 


   Arguments accepted by the jar file -    
    - The path to the noun phrases generated from articles <br>
    - The path to the vocabularies  (two separate arguments for each) <br>
    - The path to research methods results on training dataset for IDF calculation <br>
    - The path where the output files must be stored (two separate arguments for each) <br>

   It generates two files: `research_fields_results.json` and `research_methods_results.json` in `project/additional_files`, which are used to create the final output files - `methods.json` and `research_fields.json` in the `data/` folder. 




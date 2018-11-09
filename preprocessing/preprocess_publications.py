import tempfile

from preprocessing.get_input import ProcessInput
import os
import unicodedata, re
from pathlib import Path
from langdetect import detect
import enchant, spacy, json
from collections import Counter
from langdetect import DetectorFactory
DetectorFactory.seed = 0

parentPath = Path(os.getcwd()).parent
control_chars = ''.join(map(chr, list(range(0,32)) + list(range(127,160))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def remove_control_chars(s):
    return control_char_re.sub('', s)

# to get metadata for all pdfs run: "for i in *.pdf; do pdfinfo "$i" >> ../pdf-info/"$i".txt; done"

class PublicationPreprocessing:
    def __init__(self):
        self.pi = ProcessInput()
        self.pub_df = self.pi.load_publication_input()

    def write_nounPharses_to_file(self, np_dict, file_name):
        directory = os.path.join(str(parentPath), "train_test/files/noun_phrases/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+file_name, 'w') as file:
            file.write(json.dumps(np_dict, indent=4))

    def gather_nounPhrases(self, text):
        nlp = spacy.load('en')
        doc= nlp(text.replaceAll('\n', ' '))
        index = 0
        nounIndices = []
        all_phrases = []
        for token in doc:
            # print(token.text, token.pos_, token.dep_, token.head.text)
            if token.pos_ == 'NOUN':
                nounIndices.append(index)
            index = index + 1

        for idxValue in nounIndices:
            doc = nlp(text.replaceAll('\n', ' '))
            span = doc[doc[idxValue].left_edge.i: doc[idxValue].right_edge.i + 1]
            span.merge()

            for token in doc:
                if token.dep_ == 'dobj' or token.dep_ == 'pobj' or token.pos_ == "PRON":
                    if (token.text.strip().find(' ') == -1 and token.pos_ == "PRON"): #remove single words that are pronouns
                        continue
                    token.text
                    all_phrases.append(token.text)
        return all_phrases


    def remove_noise(self, fileobj, separator='\n'):
        data = fileobj.read()
        lines = data.split('\n')
        total_lines = len(lines)
        content = []
        for i in range(total_lines):
            lines[i] = remove_control_chars(lines[i])
            numbers = sum(c.isdigit() for c in lines[i])
            uppercase_count = sum(1 for c in lines[i] if c.isupper())

            if (numbers > 0.1*len(lines[i])):
                continue
            if(uppercase_count == len(lines[i])): # probably heading
                content.append(lines[i])
            elif(uppercase_count > 0.5*len(lines[i])):
                continue
            else:
                content.append(lines[i])

        return '\n'.join(content)

    def extract_paragraphs(self, doc):
        lines = doc.split('\n')
        i = 1
        one_para = []
        all_paras = []
        sentence = ''
        while(i < len(lines)):
            if(lines[i-1].endswith(('.', '!', '?')) and len(lines[i]) > 1 and lines[i][0].isupper()):
                one_para.append(lines[i-1])
                all_paras.append(' '.join(one_para))
                one_para = []

            elif(not lines[i-1].endswith(('.', '!', '?')) and len(lines[i]) > 1 and lines[i][0].islower()):
                if(lines[i-1].endswith('-')):  #handle hyphenation
                    last_word_prev_line = lines[i-1].split()[-1]
                    first_word_curr_line = lines[i].split()[0]
                    new_word=last_word_prev_line+first_word_curr_line
                    dict = enchant.Dict("en_US")
                    if(not dict.check(new_word)):
                        new_word=last_word_prev_line[:-1] + first_word_curr_line
                        lines[i-1].replace(last_word_prev_line, new_word)
                        lines[i].replace(first_word_curr_line, '')
                one_para.append(lines[i-1])

            elif(not lines[i-1].endswith(('.', '!', '?')) and len(lines[i]) > 1 and lines[i][0].isupper()): #probably a heading
                one_para.append(lines[i - 1])
                all_paras.append(' '.join(one_para))
                one_para=[]
            else:
                one_para.append(lines[i-1])
            i += 1
        one_para.append(lines[i-1])
        all_paras.append(' '.join(one_para))

        return all_paras

    def process_text(self, extract_np=True):
        i = 0
        all_abstracts = list()
        all_content = list()
        all_ref = list()
        count_ack =0
        ack_files = list()
        count_ref =0
        no_abstract_mention = 0
        no_keyword_mention= 0
        no_abstract = list()
        no_keyword = list()
        no_intro = list()
        abstract_file =list()
        all_files = list()
        nlp = spacy.load('en')
        text_file_path = os.path.join(str(parentPath), "train_test/files/text/")
        for _,row in self.pub_df.iterrows():

            all_files.append(row['pdf_file_name'])

            with open(text_file_path + row['text_file_name'], 'r') as txt_file:
                reduced_content = self.remove_noise(txt_file)

            abstract_found = False
            abstract_match = re.search('Abstract|ABSTRACT', reduced_content)
            if(abstract_match): #not None
                abstract_beg = abstract_match.start()
            else:
                abstract_beg =-1
            abstract_end = -1

            if(abstract_beg < 0):
                no_abstract_mention += 1

            keywords_match=re.search("keywords|key words|key-words", reduced_content, flags=re.IGNORECASE)
            if(keywords_match):
                keywords_beg= keywords_match.start()
            else:
                keywords_beg= -1
            if (keywords_beg < 0):
                no_keyword_mention += 1
                no_keyword.append(row['pdf_file_name'])

            if (keywords_beg > abstract_beg): #when keywords are present after the abstract
                abstract_end = keywords_beg

            introduction_match = re.search("Introduction|INTRODUCTION", reduced_content)
            if(introduction_match):
                introduction_beg = introduction_match.start()
            else:
                introduction_beg = -1
            # introduction_beg = row['text'].lower().find('introduction')
            if(introduction_beg < 0):
                no_intro.append(row['pdf_file_name'])
            elif(keywords_beg < abstract_beg < introduction_beg):
                if(abstract_beg > 0):
                    abstract_end = introduction_beg
                elif(keywords_beg > 0): #for keywords that lie above the abstract
                    abstract_beg = keywords_beg
                    abstract_end = introduction_beg
                    #abstract = reduced_content[abstract_beg:abstract_end]
                    # count = Counter(([token.pos_ for token in nlp(abstract)])) #check if it doesn't just contain deptt/author names
                    # if (count['PROPN'] > 0.68 * sum(count.values())):
                    #     abstract_beg = -1
                    # if (len(abstract.split('.')) < 2): #abstract won't be of just 1 sentence
                    #     abstract_beg = -1


            if(abstract_beg < 0): # extract the first para as abstract
                paras = self.extract_paragraphs(reduced_content)
                for p in paras:
                    try:
                        if (row['title'] in p or len(p.split(' ')) < 10 or len(p.split('.')) < 2 or detect(p) != 'en'): # less than 10 words in a para or just 1 line
                            reduced_content = reduced_content.replace(p,' ').strip()
                            continue
                        else:
                            count = Counter(([token.pos_ for token in nlp(p)]))
                            if(count['PROPN'] > 0.68 * sum(count.values())):
                                reduced_content = reduced_content.replace(p,' ').strip()
                                continue
                            else:
                                abstract = p
                                all_abstracts.append(abstract)
                                abstract_file.append(row['pdf_file_name'])
                                abstract_found= True
                                break
                    except: #raise LangDetectException(ErrorCode.CantDetectError,
                        continue

            else:
                abstract = reduced_content[abstract_beg:abstract_end]
                all_abstracts.append(abstract)
                abstract_file.append(row['pdf_file_name'])
                abstract_found= True
                reduced_content = reduced_content.replace(abstract, ' ').strip()
                if len(abstract)==0:
                    no_abstract.append(row['pdf_file_name'])

            if(not abstract_found):
                all_abstracts.append(row['title'])  #3126.pdf

            # remove references or bibliography
            references_match = re.search('references|bibliography|literature cited|notes|literature review|literature', reduced_content, flags=re.IGNORECASE)
            if(references_match):
                references_beg = references_match.start()
            else:
                references_beg = -1

            if (references_beg < 0):
                pattern = re.compile(r'^\[\d+\]\s[A-Za-z]')
                for match in re.findall(pattern, reduced_content):
                    if (len(match) < 10):
                        print(row['pdf_file_name'])
                        break
                    else:
                        ref = match
                        reduced_content = re.sub(pattern, '', reduced_content)
                        all_ref.append(ref)

            if (references_beg > 0):
                ref = row['text'][references_beg:]
                row['text'] = row['text'][:references_beg]
                all_ref.append(ref)

            # remove acknowledgement
            ack_match = re.search('acknowledgments|acknowledgment|acknowledgement|acknowledgements', reduced_content, flags=re.IGNORECASE)
            if(ack_match):
                ack_beg = ack_match.start()
            else:
                ack_beg = -1
            # TODO: handle 'thank/grateful'. can appear anywhere in the article as a part of acknowledgement. remove that para
            if ack_beg < 0:
                count_ack += 1
                ack_files.append(row['pdf_file_name'])
            if ack_beg > 0:
                reduced_content = reduced_content[:ack_beg]

            all_content.append(reduced_content)

        print(len(all_abstracts))
        # temp3 = [item for item in all_files if item not in abstract_file]
        # print(temp3)

        self.pub_df['abstract'] = all_abstracts
        self.pub_df['processed_text'] = all_content

        if(extract_np):
            for _, row in self.pub_df.iterrows():
                np_dict = dict()
                np_dict['title'] = self.gather_nounPhrases(row['title'])
                np_dict['abstract'] = self.gather_nounPhrases(row['abstract'])
                np_dict['processed_text'] = self.gather_nounPhrases(row['processed_text'])
                self.write_nounPharses_to_file(np_dict,row['text_file_name'])
                print('files written.')

        print(count_ref, count_ack)
        print(ack_files)
        print(len(all_ref), len(no_intro))
        print(no_keyword_mention, no_abstract_mention)
        print(len(no_keyword), no_keyword)
        print(len(no_abstract), no_abstract)

def main():
    obj = PublicationPreprocessing()
    obj.process_text()

if __name__ == '__main__':
    main()
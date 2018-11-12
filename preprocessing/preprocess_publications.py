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
            file.write(json.dumps(np_dict, indent=4, ensure_ascii=False))

    def write_processed_content(self, content_dict, file_name):
        directory = os.path.join(str(parentPath), "train_test/files/processed_articles/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + file_name, 'w') as file:
            file.write(json.dumps(content_dict,  indent=4, ensure_ascii=False))

    def fetch_pdf_info(self, file_name):
        """
        for every pdf, returns a dictionary containing metadata information
        :param file_name:
        :return:
        """
        print(file_name)
        info = dict()
        directory = os.path.join(str(parentPath), "train_test/files/pdf-info/")
        with open(directory + file_name + ".txt", encoding='utf-8', errors='ignore') as file:

            lines = file.readlines()
            i=0
            while (i < (len(lines))):
                k, v = lines[i].strip().split(':', 1)
                while (i+1 < len(lines) and lines[i+1].find(':') == -1):
                    v += " " + lines[i+1]
                    i += 1
                info[k.strip()] = v.strip()
                i+=1

            print('info', info)
            return info

    def gather_nounPhrases(self, text):
        nlp = spacy.load('en')
        doc= nlp(text.replace('\n', ' '))
        index = 0
        nounIndices = []
        all_phrases = []
        for token in doc:
            # print(token.text, token.pos_, token.dep_, token.head.text)
            if token.pos_ == 'NOUN':
                nounIndices.append(index)
            index = index + 1

        for idxValue in nounIndices:
            doc = nlp(text.replace('\n', ' '))
            span = doc[doc[idxValue].left_edge.i: doc[idxValue].right_edge.i + 1]
            span.merge()

            for token in doc:
                if token.dep_ == 'dobj' or token.dep_ == 'pobj' or token.pos_ == "PRON":
                    if (token.text.strip().find(' ') == -1 and token.pos_ == "PRON"): #remove single words that are pronouns
                        continue
                    if (c.isdigit() for c in token.text):
                        continue
                    print('np: ',token.text)
                    all_phrases.append(token.text.replace('\n', ' '))

        return all_phrases

    def extract_paragraphs(self, doc):
        lines = doc.split('\n')
        i = 1
        one_para = []
        all_paras = []

        while (i < len(lines)):
            if (lines[i - 1].endswith(('.', '!', '?')) and len(lines[i]) > 1 and lines[i][0].isupper()):
                one_para.append(lines[i - 1])
                all_paras.append(' '.join(one_para))
                one_para = []

            elif (not lines[i - 1].endswith(('.', '!', '?')) and len(lines[i]) > 1 and lines[i][0].islower()):
                one_para.append(lines[i - 1])

            elif (not lines[i - 1].endswith(('.', '!', '?')) and len(lines[i]) > 3 and lines[i][0].isupper()):  # probably a heading
                one_para.append(lines[i - 1])
                all_paras.append(' '.join(one_para))
                one_para = []
            else:
                one_para.append(lines[i - 1])
            i += 1
        one_para.append(lines[i - 1])
        all_paras.append(' '.join(one_para))

        return all_paras

    def remove_url(self, text):
        text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text, flags=re.MULTILINE)
        return text

    def remove_noise_and_handle_hyphenation(self, doc, dehyphenation=True):
        lines = doc.split('\n')
        i = 1
        content = []
        while (i <= len(lines)):
            lines[i-1] = remove_control_chars(lines[i-1])
            numbers = sum(c.isdigit() for c in lines[i-1])
            uppercase_count = sum(1 for c in lines[i-1] if c.isupper())
            symbol_count = sum(1 for c in lines[i-1] if c in ['(', ')', '{', '}', '*', '+', '-', '/', '=', '<', '>','%'])

            if(re.search('JEL-?Classification|JEL\s?-?[c|C]lassification|JEL-?CLASSIFICATION',lines[i-1])):
                i+=1
                continue

            if(len(lines[i-1]) <= 3): #for characters in equations
                i+=1
                continue
            if(numbers+symbol_count > 0.4*len(lines[i-1])):
                i+=1
                continue
            if (numbers > 0.4*len(lines[i-1])):
                i += 1
                continue
            if (symbol_count > 0.25 * len(lines[i - 1])):
                i += 1
                continue
            if(uppercase_count == len(lines[i-1])): # probably heading
                content.append(lines[i-1])
            elif(uppercase_count > 0.6*len(lines[i-1])):
                i += 1
                continue
            else:
                if (dehyphenation and lines[i - 1].endswith('-')):  # handle hyphenation
                    last_word_prev_line = lines[i - 1].split()[-1]
                    if(len(lines[i].split()) > 0):
                        first_word_curr_line = lines[i].split()[0]
                        new_word = last_word_prev_line + first_word_curr_line
                        dict = enchant.Dict("en_US")
                        if (not dict.check(new_word)):
                            new_word = last_word_prev_line[:-1] + first_word_curr_line
                            lines[i - 1] = lines[i - 1].replace(last_word_prev_line, new_word)
                            lines[i] = lines[i].replace(first_word_curr_line, '')
                content.append(lines[i-1])
            i+=1

        return '\n'.join(content)

    def process_text(self, extract_np=False, write_processed_files=True):
        all_abstracts = list()
        all_content = list()
        all_intro = list()
        all_methods = list()
        all_results = list()
        all_summary = list()
        all_discussions = list()
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
        content_dict = dict()
        all_keywords = list()
        all_subjects = list()
        no_methods = list()
        nlp = spacy.load('en')
        for _,row in self.pub_df.iterrows():

            pdf_info = self.fetch_pdf_info(row['pdf_file_name'])

            reduced_content = row['text']

            # remove references or bibliography
            references_match = re.search(
                'References:?|Bibliography:?|Literature Cited:?|Notes:?|Literature:?|REFERENCES:?|BIBLIOGRAPHY:?|LITERATURE CITED:?|NOTES:?|LITERATURE:?',
                reduced_content)

            if (references_match):
                references_beg = references_match.start()
                ref = reduced_content[references_beg:]
                reduced_content = reduced_content[:references_beg]
                all_ref.append(ref)
            else:
                references_beg = -1

            if (references_beg < 0):
                pattern = re.compile(r'^\[\d+\]\s[A-Za-z]')
                for match in re.findall(pattern, reduced_content):
                    if (len(match) < 10):
                        print(match)
                        print(row['pdf_file_name'])
                        break
                    else:
                        ref = match
                        reduced_content = re.sub(pattern, '', reduced_content)
                        all_ref.append(ref)

            # remove acknowledgement
            ack_match = re.search('Acknowledge?ments?|ACKNOWLEDGE?MENTS?', reduced_content)
            if (ack_match):
                ack_beg = ack_match.start()
                reduced_content = reduced_content[:ack_beg]
            else:
                ack_beg = -1  # TODO: handle 'thank/grateful'. can appear anywhere in the article as a part of acknowledgement. remove that para
            if ack_beg < 0:
                count_ack += 1
                ack_files.append(row['pdf_file_name'])

            reduced_content = reduced_content.replace(row['title'], '')

            reduced_content = self.remove_noise_and_handle_hyphenation(reduced_content, dehyphenation=True)
            reduced_content = self.remove_url(reduced_content)
            # tables_index = [m.start() for m in re.finditer('Table|TABLE', reduced_content)]
            # for i in tables_index:
            #     selected_content = reduced_content[i:]
            #     lines = selected_content.split('\n')
            #     for i in range(len(lines)):
            #         symbol_count = sum(
            #             1 for c in lines[i - 1] if c in ['(', ')', '{', '}', '*', '+', '-', '/', '=', '<', '>'])
            #         if 'Table' in lines[i] or 'TABLE' in lines[i]:
            #             reduced_content = reduced_content.replace(lines[i], '')
            #         elif len(lines[i].split()) <= 7:
            #             reduced_content = reduced_content.replace(lines[i], '')
            #         elif(symbol_count > 0.15*len(lines[i])):
            #             reduced_content = reduced_content.replace(lines[i], '')
            #         else:
            #             break

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
                keywords = reduced_content[keywords_beg:]
                start = keywords.find(':')
                keywords = keywords[start+1:].strip()
                keywords = keywords.replace(', \n', ', ').replace('-\n', '')
                keywords = keywords.split('\n')[0]
                keyword_list = re.split(',|;',keywords)
                print('keywords', keyword_list)

                if 'Keywords' in pdf_info.keys():
                    keyword_list = re.split(',|;', pdf_info['Keywords'])

                content_dict['keywords'] = keyword_list

                all_keywords.append(keyword_list)
            else:
                keywords_beg= -1

            if (keywords_beg < 0):
                content_dict['keywords'] = []
                no_keyword_mention += 1
                all_keywords.append(list())
                print('no keywords in: ', row['pdf_file_name'])
                no_keyword.append(row['pdf_file_name'])

            if (keywords_beg > abstract_beg): #when keywords are present after the abstract
                abstract_end = keywords_beg

            introduction_match = re.search("Introduction|INTRODUCTION", reduced_content)
            if(introduction_match):
                introduction_beg = introduction_match.start()
            else:
                introduction_beg = -1

            if(introduction_beg < 0):
                no_intro.append(row['pdf_file_name'])

            elif(keywords_beg < abstract_beg < introduction_beg):
                if(abstract_beg > 0):
                    abstract_end = introduction_beg
                elif(keywords_beg > 0): #for keywords that lie above the abstract
                    abstract_beg = keywords_beg
                    abstract_end = introduction_beg

            if(abstract_beg < 0): # extract the first para as abstract
                paras = self.extract_paragraphs(reduced_content)
                for p in paras:
                    try:
                        if (len(p.split(' ')) < 10 or len(p.split('.')) < 2 or detect(p) != 'en'): # less than 10 words in a para or just 1 line
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
                                content_dict['abstract'] = abstract
                                abstract_file.append(row['pdf_file_name'])
                                abstract_found= True
                                break
                    except: #raise LangDetectException(ErrorCode.CantDetectError,
                        continue

            else:
                abstract = reduced_content[abstract_beg:abstract_end]
                content_dict['abstract'] = abstract
                all_abstracts.append(abstract)
                abstract_file.append(row['pdf_file_name'])
                abstract_found= True
                reduced_content = reduced_content.replace(abstract, ' ').strip()
                if len(abstract)==0:
                    no_abstract.append(row['pdf_file_name'])

            if(not abstract_found):
                content_dict['abstract'] = row['title']
                all_abstracts.append(row['title'])  #3126.pdf

            if 'Subject' in pdf_info:
                all_subjects.append(pdf_info['Subject'])
                content_dict['subject'] = pdf_info['Subject']
            else:
                all_subjects.append('')
                content_dict['subject'] = ''

            for key in pdf_info:
                if pdf_info[key] in reduced_content:
                    print(pdf_info[key], reduced_content.find(pdf_info[key]), len(reduced_content))
                    reduced_content = reduced_content.replace(pdf_info[key], ' ')
            print(len(reduced_content))

            methods_index = [m.start() for m in
                             re.finditer('Methodology|Methods?|METHODS?|METHODOLODY|Data|DATA', reduced_content)]
            summary_index = [m.start() for m in re.finditer('Conclusion|Summary|CONCLUSION|SUMMARY', reduced_content)]
            results_index = [m.start() for m in re.finditer('Results?|RESULTS?', reduced_content)]
            discussion_index = [m.start() for m in re.finditer('Discussions?|DISCUSSIONS?', reduced_content)]

            methods_beg = -1
            results_beg = -1
            summary_beg = -1
            discussion_beg = -1
            if len(results_index) > 0:
                results_beg = results_index[-1]
            if len(summary_index) > 0:
                summary_beg = summary_index[-1]
            if len(discussion_index) > 0:
                discussion_beg = discussion_index[-1]
            if(introduction_beg > 0):
                intro_found = False
                for m in methods_index:
                    if (m > introduction_beg):
                        methods_beg = m
                        intro = reduced_content[introduction_beg:methods_beg]
                        intro_found = True
                        reduced_content = reduced_content.replace(intro, ' ').strip()
                        all_intro.append(intro)
                        break
                if(not intro_found):
                    all_intro.append('')

            if methods_beg > 0:
                if results_beg > 0:
                    methodology = reduced_content[methods_beg:results_beg]
                elif summary_beg > 0 and summary_beg < discussion_beg:
                    methodology = reduced_content[methods_beg:summary_beg]
                elif discussion_beg > 0:
                    methodology = reduced_content[methods_beg:discussion_beg]
                else:
                    methodology = reduced_content[methods_beg:]
                all_methods.append(methodology)
                content_dict['methodogy'] = methodology
                reduced_content = reduced_content.replace(methodology, ' ').strip()
            else:
                all_methods.append('')
                content_dict['methodology'] = ''
                no_methods.append(row['pdf_file_name'])

            if summary_beg > 0:
                if summary_beg < discussion_beg:
                    summary = reduced_content[summary_beg:discussion_beg]
                else:
                    summary = reduced_content[summary_beg:]
                all_summary.append(summary)
                reduced_content.replace(summary, ' ').strip()
                content_dict['summary'] = summary
            else:
                content_dict['summary'] = ''
                all_summary.append('')


            content_dict['reduced_content'] = reduced_content
            all_content.append(reduced_content)

            if(write_processed_files):
                self.write_processed_content(content_dict, row['text_file_name'])

        print(len(all_abstracts))
        print(len(no_methods), no_methods)
        # temp3 = [item for item in all_files if item not in abstract_file]
        # print(temp3)

        self.pub_df['abstract'] = all_abstracts
        self.pub_df['processed_text'] = all_content
        self.pub_df['keywords'] = all_keywords
        self.pub_df['subject'] = all_subjects
        self.pub_df['methodology'] = all_methods
        self.pub_df['summary'] = all_summary

        if(extract_np):
            for _, row in self.pub_df.iterrows():
                np_dict = dict()
                np_dict['title'] = self.gather_nounPhrases(row['title'])
                print(np_dict)
                np_dict['abstract'] = self.gather_nounPhrases(row['abstract'])
                np_dict['processed_text'] = self.gather_nounPhrases(row['processed_text'])
                np_dict['keywords'] = self.gather_nounPhrases(row['keywords'])
                self.write_nounPharses_to_file(np_dict,row['text_file_name'])
                print('files written.')
                break

        print(count_ref, count_ack)
        print(ack_files)
        print(len(all_ref), len(no_intro))
        print(no_keyword_mention, no_abstract_mention)
        print(len(no_keyword), no_keyword)
        print(len(no_abstract), no_abstract)

        return self.pub_df


def main():
    obj = PublicationPreprocessing()
    obj.process_text(extract_np=False, write_processed_files=True)

if __name__ == '__main__':
    main()
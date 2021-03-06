import tempfile

from project.preprocessing.get_input import ProcessInput
import os, subprocess
import re, logging
from pathlib import Path
from langdetect import detect
import enchant, spacy, json
from collections import Counter
from timeit import default_timer as timer
from langdetect import DetectorFactory
DetectorFactory.seed = 0


control_chars = ''.join(map(chr, list(range(0,32)) + list(range(127,160))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))


def remove_control_chars(s):
    return control_char_re.sub('', s)


class PublicationPreprocessing:
    def __init__(self, path='data/input/'):
        self.nlp = spacy.load('en', disable=['ner', 'textcat'])
        self.pi = ProcessInput()
        self.pub_df = self.pi.load_publication_input(path=path)

    def write_nounPharses_to_file(self, np_dict, file_name):
        """
        Writes the dict to a file on the disk.
        :param np_dict: a dict containing noun phrases present in the relevant sections of a publication
                        key -> relevant section (eg. abstract) value -> noun phrases present in that section
        :param file_name:
        :return:
        """
        directory = "project/additional_files/nounPhrases/"
        self.create_directory(directory)
        # os.chmod("project/additional_files/", 0o777)
        # os.chmod(directory, 0o777)

        with open(directory + file_name, 'w+') as file:
            json.dump(np_dict, file, indent=4, ensure_ascii=False)

    def write_processed_content(self, content_dict, file_name):
        """
        Writes the dict to a file on the disk.
        :param content_dict: contains preprocessed/denoised text of the relevant sections of a pub
                            key -> relevant section (eg. abstract) value -> denoised text in that section
        :param file_name:
        :return:
        """
        directory = "project/additional_files/processed_articles/"
        self.create_directory(directory)
        with open(directory + file_name, 'w+') as file:
            json.dump(content_dict, file, indent=4, ensure_ascii=False)

    def fetch_pdf_info(self, file_name):
        """
        for every pdf, returns a dictionary containing metadata information
        :param file_name:
        :return:
        """
        info = dict()
        directory = "project/additional_files/pdf-info/"

        with open(directory + file_name, encoding='utf-8', errors='ignore') as file:

            lines = file.readlines()
            i=0
            while (i < (len(lines))):
                k, v = lines[i].strip().split(':', 1)
                while (i+1 < len(lines) and lines[i+1].find(':') == -1):
                    v += " " + lines[i+1]
                    i += 1
                info[k.strip()] = v.strip()
                i+=1

            return info

    def extract_text_from_pdf(self, pdf_name, txt_name):
        """
        Converts a PDF into text using the open source tool 'pdftotext' by Poppler Utils, writes the text file to disk
        :param file_name:
        :return: returns the converted text
        """
        directory = "project/additional_files/text/"
        self.create_directory(directory)

        subprocess.call(['pdftotext -nopgbrk "{0}" "{1}"'.format('data/input/files/pdf/' + pdf_name,
                                                         directory + txt_name)], shell=True)
        with open(directory + txt_name, 'r+', errors='ignore') as txt_file:
            data = txt_file.read()
            return data

    def gather_nounPhrases(self, text):
        """
        :param text:
        :return: noun phrases present in the given text
        """

        text = text.replace('-\n', '').replace('\n', ' ')
        doc = self.nlp(text)
        all_phrases = []

        for np in doc.noun_chunks:
            if len(str(np).split()) == 1 and all('-PRON-' in token.lemma_ for token in np) == True or str(
                    np) in all_phrases:
                continue
            all_phrases.append(str(np))

        return all_phrases

    def extract_paragraphs(self, doc):
        """
        given a text, tries to find the lines that belong to the same paragraph (e.g. abstract) and return it.
        :param doc:
        :return:
        """
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
        """
        :param text:
        :return: text devoid of url  
        """
        text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text, flags=re.MULTILINE)
        return text


    def extract_jel_methods_and_fields(self, lines, line_num, jel_method, jel_field):
        with open('project/train_test/jel-dict.json', 'r+') as file:
            jel_dict = json.load(file)

            jel_match = re.search(
                'JEL-?Classification:?\s?|JEL\s?-?[c|C]lassification:?\s?|JEL-?CLASSIFICATION:?\s?|'
                'JEL\s?-?\s?[c|C][o|O][d|D][e|E]:?\s?|JEL:?\s?', lines[line_num - 1])
            if jel_match:  # JEL-classification code found
                if lines[line_num - 1][jel_match.end() - 1] == ':':  # because lines have already been split based on newline,
                    line_num += 1

                matches = re.finditer(r"[a-zA-Z]\d+", lines[line_num - 1],
                                      re.MULTILINE)  # JEL-code contains 1 alphabet and 2-3 digits

                for match in matches:
                    code = lines[line_num - 1][match.start():match.end()][:3]  # take only the first three characters and get
                    # the corresponding term from jel-dictionary
                    if code in jel_dict:
                        if code.startswith('c') or code.startswith('C'):
                            jel_method.append(jel_dict[code])
                        else:
                            jel_field.append(jel_dict[code])
                return True
            return False

    def remove_noise_handle_hyphenation_extract_jel_codes(self, doc, dehyphenation=True):
        """
        :param doc: given text
        :param dehyphenation: boolean value
        :return: 1) text without unnecessary hyphens and irrelevant data, 2) JEL-fields and 3) JEL-methods
        (if JEL-classification codes are found)
        """
        content = []
        jel_method = []
        jel_field = []

        lines = doc.split('\n')
        i = 1

        # with open('project/train_test/jel-dict.json', 'r+') as file:
        #     jel_dict = json.load(file)

        while (i <= len(lines)):
            lines[i-1] = remove_control_chars(lines[i-1]) # remove non-printable characters

            # -- extract jel fields and methods --
            if self.extract_jel_methods_and_fields(lines,i,jel_method, jel_field): # if jel-match found
                i+=1
                continue

            # ---- identify and remove irrelevant data ----
            numbers_count = sum(c.isdigit() for c in lines[i-1])
            uppercase_count = sum(1 for c in lines[i-1] if c.isupper())
            symbol_count = sum(1 for c in lines[i-1] if c in ['(', ')', '{', '}', '*', '+', '-', '/', '=', '<', '>','%','.'])

            # NOTE: all the threshold values have been set after trial and testing.
            # -> Run test_preprocess_publications.py for unit-testing

            if(len(lines[i-1]) <= 3): # for characters in equations
                i+=1
                continue
            if(numbers_count+symbol_count > 0.28*len(lines[i-1])):
                i+=1
                continue
            if (numbers_count > 0.2*len(lines[i-1])):
                i += 1
                continue
            if (symbol_count > 0.25 * len(lines[i - 1])):
                i += 1
                continue
            if(uppercase_count > 0.7*len(lines[i-1])): # probably heading, so include in the final content
                content.append(lines[i-1])

            else:
                if (dehyphenation and lines[i - 1].endswith('-')):  # handle hyphenation
                    last_word_prev_line = lines[i - 1].split()[-1]
                    if(len(lines[i].split()) > 0):
                        first_word_curr_line = lines[i].split()[0]
                        new_word = last_word_prev_line + first_word_curr_line
                        dict = enchant.Dict("en_US") # check the word 
                        if (not dict.check(new_word)):
                            new_word = last_word_prev_line[:-1] + first_word_curr_line
                        lines[i - 1] = lines[i - 1].replace(last_word_prev_line, new_word)
                        lines[i] = lines[i].replace(first_word_curr_line, '')
                content.append(lines[i-1])
            i+=1

        return '\n'.join(content), jel_field, jel_method

    def process_text(self, extract_np=True, write_processed_files=True):

        content_dict = dict()
        no_method_match = [] # -- pubs with no regex match for methodology
        no_method_found = [] # -- pubs with no 'methodology' section in content_dict
        no_abstract = [] # -- pubs with no abstract
        no_keywords = [] # -- pubs with no keywords


        directory = "project/additional_files/pdf-info/"
        self.create_directory(directory)

        for _,row in self.pub_df.iterrows():
            try:
                np_file = Path('project/additional_files/nounPhrases/', row['text_file_name'])

                if np_file.exists():
                    continue
                else:
                    print(row['pdf_file_name'])
                    start1 = timer()

                    # -- extract pdf metadata
                    pdf_info = self.extract_pdf_metadata(row)

                    # -- extract text from pdf
                    row['text'] = self.extract_text_from_pdf(row['pdf_file_name'], row['text_file_name'])
                    reduced_content = row['text']

                    # remove references or bibliography
                    reduced_content = self.remove_references(reduced_content)

                    # remove acknowledgement
                    reduced_content = self.remove_acknowledgment(reduced_content)

                    # -- clean up remaining text
                    reduced_content = reduced_content.replace(row['title'], '')

                    reduced_content, jel_field, jel_method = self.remove_noise_handle_hyphenation_extract_jel_codes(reduced_content, dehyphenation=True)

                    # --- store jel fields and methods in content dict
                    self.store_jel_fields_methods(content_dict, jel_field, jel_method, row)

                    reduced_content = self.remove_url(reduced_content)

                    # --- extract main sections from remaining content
                    abstract_found = False
                    abstract_beg = self.find_abstract(reduced_content)
                    abstract_end = -1

                    # --- find_keywords
                    abstract_beg, abstract_end, keywords_beg = self.find_keywords(abstract_beg, abstract_end,
                                                                                  content_dict, no_keywords, pdf_info,
                                                                                  reduced_content, row)
                    # --- find introdcution
                    introduction_beg = self.find_introduction(reduced_content)

                    # --- find abstract
                    if keywords_beg < abstract_beg < introduction_beg:
                        abstract_end = introduction_beg

                    elif abstract_beg < introduction_beg and keywords_beg < 0:
                        abstract_end = introduction_beg
                        if abstract_beg < 0:
                            abstract_beg = 0

                    elif introduction_beg < 0 and keywords_beg < 0 and abstract_beg < 0:
                        abstract_beg = 0

                    if abstract_beg <= 0: # extract the first para as abstract
                        paras = self.extract_paragraphs(reduced_content)
                        for p in paras:
                            try:
                                if (len(p.split(' ')) < 10 or len(p.split('.')) < 2 or detect(p) != 'en'): # less than 10 words in a para or just 1 line
                                    reduced_content = reduced_content.replace(p,'').strip()
                                    continue
                                if p in list(pdf_info.values()):
                                    reduced_content = reduced_content.replace(p, '').strip()
                                    continue
                                else:
                                    count = Counter(([token.pos_ for token in self.nlp(p)]))
                                    if(count['PROPN'] > 0.68 * sum(count.values())):
                                        reduced_content = reduced_content.replace(p,'').strip()
                                        continue
                                    else:
                                        abstract = p
                                        content_dict['abstract'] = abstract
                                        abstract_found= True
                                        reduced_content = reduced_content.replace(abstract, '').strip()
                                        break
                            except: #raise LangDetectException(ErrorCode.CantDetectError,
                                continue

                    else: #if abstract_end == keywords_beg or intro_beg
                        if abstract_end > 0:
                            abstract = reduced_content[abstract_beg:abstract_end]
                            content_dict['abstract'] = abstract
                            abstract_found= True
                            reduced_content = reduced_content[abstract_end:]

                    # -- find the subject of article
                    self.find_subject_of_article(content_dict, pdf_info)

                    # -- remove metadata from article
                    reduced_content = self.remove_all_metadata_content_from_article(pdf_info, reduced_content)

                    # -- find all other sections
                    methods_match = [(m.start(), m.end()) for m in re.finditer(r"^.*(Data|DATA|Methodology|\b[m|M]ethods?\b|METHODS?|METHODOLODY|APPROACH|Approach).*$", reduced_content, flags=re.MULTILINE)]
                    summary_index = [m.start() for m in re.finditer(r"^\d?\.?\s?Summary|^\d?\.?\s?SUMMARY|^\d?\.?\s?Conclusions?|^\d?\.?\s?CONCLUSIONS?|^\d?\.?\s?Concluding\sRemarks|^\d?\.?\s?CONCLUDING\sREMARKS", reduced_content, flags=re.MULTILINE)]
                    results_index = [m.start() for m in re.finditer(r"^\d?\.?\s?Results?|^\d?\.?\s?RESULTS?", reduced_content, flags=re.MULTILINE)]
                    discussion_index = [m.start() for m in re.finditer(r"^\d?\.?\s?Discussions?|^\d?\.?\s?DISCUSSIONS?", reduced_content, flags=re.MULTILINE)]
                    methods_beg = -1
                    results_beg = -1
                    summary_beg = -1
                    discussion_beg = -1
                    intro_found = False

                    if len(methods_match) == 0:
                        no_method_match.append(row['pdf_file_name'])

                    # -- find introduction
                    introduction_beg = self.find_introduction(reduced_content)
                    if len(results_index) > 0:
                        results_beg = results_index[-1] # -- the heading can also appear in table of contents, so take the last match found
                    if len(summary_index) > 0:
                        summary_beg = summary_index[-1]
                    if len(discussion_index) > 0:
                        discussion_beg = discussion_index[-1]
                        # Discussion papers
                        if discussion_beg < results_beg or discussion_beg < abstract_beg or \
                                discussion_beg < keywords_beg or discussion_beg < introduction_beg:
                            discussion_beg = -1   # discussion section not found yet

                    # -- find methodology
                    methods_beg = self.find_introduction_section(content_dict, intro_found, introduction_beg,
                                                                 methods_beg, methods_match, reduced_content)

                    self.find_methodology_section(content_dict, discussion_beg, methods_beg, no_method_found,
                                                  reduced_content, results_beg, row['pdf_file_name'], summary_beg)

                    # -- find abstract
                    self.find_abstract_section(abstract_beg, abstract_end, abstract_found, content_dict, methods_beg,
                                               no_abstract, reduced_content, row)

                    # -- find summary and discussion sections
                    reduced_content = self.find_summary_discussion_sections(content_dict, discussion_beg,
                                                                            reduced_content, summary_beg)

                    # -- remove the sections already found so that we search through the remaining content
                    reduced_content = reduced_content.replace(content_dict['introduction'], '').strip()
                    reduced_content = reduced_content.replace(content_dict['methodology'], '').strip()

                    if reduced_content.find(content_dict['abstract']) > 0:
                        reduced_content = reduced_content[reduced_content.find(content_dict['abstract']) + len(content_dict['abstract']):]

                    # -- reduced content stores all the sections not found
                    content_dict['reduced_content'] = reduced_content

                    # -- write the content_dict to file
                    if(write_processed_files):
                        self.write_processed_content(content_dict, row['text_file_name'])

                    # -- extract noun-phrases from the content_dict that contains section-wise data
                    if (extract_np):
                        self.get_section_wise_NPs(content_dict, row)

            except Exception as e:
                logging.exception(e)
                continue

        print(f'no methods match: {len(no_method_match)} , {no_method_match}')  # -- pubs with no regex match for methodology
        print(f'no methods found: {len(no_method_found)} , {no_method_found}')  # -- pubs with no 'methodology' section in content_dict
        print(f'no abstract found: {len(no_abstract)} , {no_abstract}')  # -- pubs with no abstract
        print(f'no methods found: {len(no_keywords)} , {no_keywords}')  # -- pubs with no keywords

    def get_section_wise_NPs(self, content_dict, df_row):
        np_dict = dict()
        np_dict['title'] = self.gather_nounPhrases(df_row['title'])

        if len(content_dict['abstract']) > 0:
            np_dict['abstract'] = self.gather_nounPhrases(content_dict['abstract'])

        if (len(content_dict['abstract'].split()) < 25 and len(content_dict['methodology'].split()) < 20):
            np_dict['processed_text'] = self.gather_nounPhrases(content_dict['reduced_content'])

        np_dict['keywords'] = self.gather_nounPhrases('; '.join(content_dict['keywords']))
        if (len(content_dict['abstract'].split()) < 25):
            np_dict['summary'] = self.gather_nounPhrases(content_dict['summary'])

        np_dict['methodology'] = self.gather_nounPhrases(content_dict['methodology'])
        np_dict['subject'] = self.gather_nounPhrases(content_dict['subject'])
        np_dict['jel_method'] = self.gather_nounPhrases('; '.join(content_dict['jel_method']))
        np_dict['jel_field'] = self.gather_nounPhrases('; '.join(content_dict['jel_field']))

        self.write_nounPharses_to_file(np_dict, df_row['text_file_name'])

    def find_summary_discussion_sections(self, content_dict, discussion_beg, reduced_content, summary_beg):
        if summary_beg > 0:
            if summary_beg < discussion_beg:
                summary = reduced_content[summary_beg:discussion_beg]
                discussion = reduced_content[discussion_beg:]
                content_dict['discussion'] = discussion
                reduced_content = reduced_content.replace(discussion, '').strip()
            else:
                summary = reduced_content[summary_beg:]
                if discussion_beg > 0:
                    discussion = reduced_content[discussion_beg:summary_beg]
                    content_dict['discussion'] = discussion
                    reduced_content = reduced_content.replace(discussion, '').strip()

            reduced_content = reduced_content.replace(summary, '').strip()
            content_dict['summary'] = summary
        else:
            content_dict['summary'] = ''
            if discussion_beg > 0:
                discussion = reduced_content[discussion_beg:]
                content_dict['discussion'] = discussion
                reduced_content = reduced_content.replace(discussion, '').strip()
        return reduced_content

    def find_abstract_section(self, abstract_beg, abstract_end, abstract_found, content_dict, methods_beg, no_abstract,
                              reduced_content, row):
        '''
        places abstract section in content_dict
        :param abstract_beg:
        :param abstract_end:
        :param abstract_found:
        :param content_dict:
        :param methods_beg:
        :param no_abstract:
        :param reduced_content:
        :param row:
        :return:
        '''
        if abstract_beg > 0 and abstract_end == -1 and not abstract_found:
            if methods_beg > 0:
                abstract_beg = self.find_abstract(reduced_content)
                abstract = reduced_content[abstract_beg:methods_beg]
                content_dict['abstract'] = abstract
                abstract_found = True
        if not abstract_found:
            content_dict['abstract'] = row['title']
        if len(content_dict['abstract']) == 0:
            no_abstract.append(row['pdf_file_name'])

    def find_methodology_section(self, content_dict, discussion_beg, methods_beg, no_method_found, reduced_content,
                                 results_beg, pdf_file_name, summary_beg):
        '''
        places methodology section in content_dict
        :param content_dict:
        :param discussion_beg:
        :param methods_beg:
        :param no_method_found:
        :param reduced_content:
        :param results_beg:
        :param pdf_file_name:
        :param summary_beg:
        :return:
        '''
        if methods_beg > 0:
            if results_beg > 0 and results_beg > methods_beg:
                methodology = reduced_content[methods_beg:results_beg]

            elif summary_beg > 0 and summary_beg < discussion_beg:
                methodology = reduced_content[methods_beg:summary_beg]

            elif summary_beg > 0:
                methodology = reduced_content[methods_beg:summary_beg]

            elif discussion_beg > 0 and discussion_beg > methods_beg:
                methodology = reduced_content[methods_beg:discussion_beg]

            else:
                methodology = reduced_content[methods_beg:]

            content_dict['methodology'] = methodology
        else:
            content_dict['methodology'] = ''
            no_method_found.append(pdf_file_name)

    def find_introduction_section(self, content_dict, intro_found, introduction_beg, methods_beg, methods_match,
                                  reduced_content):
        """
        Places introduction section in content_dict and returns the beginning of methods section
        :param content_dict:
        :param intro_found:
        :param introduction_beg:
        :param methods_beg:
        :param methods_match:
        :param reduced_content:
        :return:
        """
        if introduction_beg > 0:
            for m in methods_match:
                m_beg = m[0] - 1
                m_end = m[1] + 1

                if (m_end > introduction_beg and m_end < len(reduced_content) and reduced_content[m_beg:m_end].find(
                        '\n') == 0 and
                        reduced_content[m_beg:m_end].rfind('\n') == m_end - m_beg - 1 and
                        sum(c.isdigit() for c in reduced_content[m_beg:m_end]) < 2 and len(
                            reduced_content[m_beg:m_end].split()) < 6):
                    methods_beg = m[0]
                    intro = reduced_content[introduction_beg:methods_beg]
                    intro_found = True
                    content_dict['introduction'] = intro
                    break
        if not intro_found:
            if methods_beg > 0:
                intro = reduced_content[:methods_beg]
                intro = intro.replace(content_dict['abstract'], '').strip()
                content_dict['introduction'] = intro

            elif methods_beg < 0 and len(methods_match) > 0:
                for m in methods_match:
                    m_beg = m[0] - 1
                    m_end = m[1] + 1

                    if (m_end < len(reduced_content) and reduced_content[m_beg:m_end].find('\n') == 0 and
                            reduced_content[m_beg:m_end].rfind('\n') == m_end - m_beg - 1 and
                            sum(c.isdigit() for c in reduced_content[m_beg:m_end]) < 2 and len(
                                reduced_content[m_beg:m_end].split()) < 6):
                        methods_beg = m[0]
                        intro = reduced_content[:methods_beg]
                        content_dict['introduction'] = intro
                        break

            else:
                content_dict['introduction'] = ''
        return methods_beg

    def remove_all_metadata_content_from_article(self, pdf_info, reduced_content):
        for key in pdf_info:
            if len(pdf_info[key]) > 3 and pdf_info[key] in reduced_content:
                reduced_content = reduced_content.replace(pdf_info[key], ' ')
        return reduced_content

    def find_subject_of_article(self, content_dict, pdf_info):
        if 'Subject' in pdf_info:
            content_dict['subject'] = pdf_info['Subject']
        else:
            content_dict['subject'] = ''

    def find_keywords(self, abstract_beg, abstract_end, content_dict, no_keywords, pdf_info, reduced_content, row):
        keywords_match = re.search("^key-?\s?words:?.?\n?\s?", reduced_content, flags=re.IGNORECASE | re.MULTILINE)
        if keywords_match:
            keywords_beg = keywords_match.start()
            keywords_end = keywords_match.end()
            keywords = reduced_content[keywords_end:]

            # print(row['text_file_name'])
            keywords = keywords.replace(', \n', ', ').replace('-\n', '')
            keywords = keywords.split('\n')[0]
            # print(keywords)
            keyword_list = re.split(',|;|/|—', keywords)
            # print('keywords', keyword_list)

            if 'Keywords' in pdf_info.keys() and len(pdf_info['Keywords']) > len(keyword_list):
                keyword_list = re.split(',|;|/|—', pdf_info['Keywords'])

            content_dict['keywords'] = keyword_list

        else:
            keywords_beg = -1
            if 'Keywords' in pdf_info.keys():
                keyword_list = re.split(',|;|/|—', pdf_info['Keywords'])
                content_dict['keywords'] = keyword_list
            else:
                content_dict['keywords'] = []
                no_keywords.append(row['pdf_file_name'])
        if keywords_beg > abstract_beg:  # when keywords are present after the abstract
            abstract_end = keywords_beg
            if abstract_beg < 0:
                abstract_beg = 0
        return abstract_beg, abstract_end, keywords_beg

    def store_jel_fields_methods(self, content_dict, jel_field, jel_method, row):
        if len(row['jel_field']) > len(jel_field) or len(row['jel_method']) > len(jel_method):
            content_dict['jel_field'] = row['jel_field']
            content_dict['jel_method'] = row['jel_method']
        else:
            content_dict['jel_field'] = jel_field
            row['jel_field'] = jel_field
            content_dict['jel_method'] = jel_method
            row['jel_method'] = jel_method

    def extract_pdf_metadata(self, df_row):
        subprocess.call(['pdfinfo "{0}" >> "{1}"'.format('data/input/files/pdf/' + df_row['pdf_file_name'],
                                                         'project/additional_files/pdf-info/' + df_row[
                                                             'text_file_name'])],
                        shell=True)
        pdf_info = self.fetch_pdf_info(df_row['text_file_name'])
        return pdf_info

    def create_directory(self, directory):
        if not os.path.exists(directory):
            oldmask = os.umask(000)
            os.makedirs(directory, mode=0o777)
            os.umask(oldmask)

    def find_introduction(self, reduced_content):
        introduction_index = [m.start() for m in
                              re.finditer(r"^\d?\.?\s?Introduction|^\d?\.?\s?INTRODUCTION", reduced_content,
                                          flags=re.MULTILINE)]
        if len(introduction_index) > 0:
            introduction_beg = introduction_index[-1]
        else:
            introduction_beg = -1
        return introduction_beg

    def find_abstract(self, reduced_content):
        abstract_match = re.search('^\d?\.?\s?Abstract:?|^\d?\.?\s?ABSTRACT:?|^\d?\.?\s?abstract:?', reduced_content,
                                   flags=re.MULTILINE)
        if (abstract_match):  # not None
            abstract_beg = abstract_match.start()
        else:
            abstract_beg = -1
        return abstract_beg

    def remove_acknowledgment(self, reduced_content):
        # TODO: handle 'thank/grateful'. can appear anywhere in the article as a part of acknowledgement. remove that para

        ack_index = [m.start() for m in re.finditer('^Acknowledge?ments?|^ACKNOWLEDGE?MENTS?', reduced_content, flags=re.MULTILINE)]

        if len(ack_index) > 0:
            ack_beg = ack_index[-1]
            reduced_content = reduced_content[:ack_beg]

        return reduced_content

    def remove_references(self, reduced_content):
        references_index = [m.start() for m in re.finditer(
            '^\d?\.?\s?References:?|^\d?\.?\s?Bibliography:?|^\d?\.?\s?Literature Cited:?|^\d?\.?\s?Notes:?|^\d?\.?\s?REFERENCES:?|^\d?\.?\s?BIBLIOGRAPHY:?|^\d?\.?\s?LITERATURE CITED:?|^\d?\.?\s?NOTES:?|^\d?\.?\s?LITERATURE:?',
            reduced_content, flags=re.MULTILINE)]

        if len(references_index) > 0:
            references_beg = references_index[-1]
            reduced_content = reduced_content[:references_beg]

        else:
            references_beg = -1
        if (references_beg < 0):
            pattern = re.compile(r'^\[\d+\]\s[\w\s].+')
            for match in re.findall(pattern, reduced_content):
                if (len(match) < 25):
                    continue
                else:
                    reduced_content = re.sub(pattern, '', reduced_content)
        return reduced_content


def main():
    obj = PublicationPreprocessing()
    obj.process_text(extract_np=True, write_processed_files=True)


if __name__ == '__main__':
    main()

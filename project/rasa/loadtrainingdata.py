# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:04:58 2018

@author: Jan
"""

import json
import re
import en_core_web_sm
import configparser

nlp = en_core_web_sm.load()
config = configparser.RawConfigParser()
config.read('conf/conf.cnf')
def build_example(entity_mentions):
    with open(config.get('conf','publication_text_folder')+str(entity_mentions[0]['publication_id'])+'.txt', 'r', encoding='utf-8') as myfile:
        textfile=myfile.read().replace('-\n','').replace('\n', ' ')
    doc = nlp(textfile)
    sentences = doc.sents
    entities=[]
    for sent in sentences:
        text=str(sent)
        for mention_list in entity_mentions:
            for mention in mention_list['mention_list']:
                #print('__MENTION__'+mention)
                #print(mention_list)
                findings= [match.start() for match in re.finditer(re.escape(mention.lower()), text.lower())]
                if len(findings)>0:
                    for finding in findings:
                        startindex=finding
                        endindex=finding+len(mention)
                        entity=str('data_set')
                        value=mention
                        entities.append({'start':startindex,'end':endindex,'entity':entity,'value':value})
        if len(entities)>0:
            common_examples.append({'text':text,'intent':'dataset_mention','entities':entities})
        entities=[]

with open(config.get('conf','data_set_citations')) as f:
    data = json.load(f)
last_publication=""
common_examples=[]

entity_mentions=[]
for citation in data:
    if(last_publication==citation['publication_id']):
        entity_mentions.append(citation)
    else:
        if len(entity_mentions) >0:
            build_example(entity_mentions)
        last_publication=citation['publication_id']
        entity_mentions=[]
        entity_mentions.append(citation)
data={'rasa_nlu_data':{'common_examples':common_examples}}
with open(config.get('conf','trainingfile'), "w",encoding='utf-8') as myfile:
    myfile.write(json.dumps(data, ensure_ascii=False))

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:12:13 2018

@author: Jan
"""

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
import configparser

conf = configparser.RawConfigParser()
conf.read('conf/conf.cnf')
training_data = load_data(conf.get('conf','trainingfile'))
trainer = Trainer(config.load(conf.get('conf','rasa_conf')))
trainer.train(training_data)
model_directory = trainer.persist(conf.get('conf','modelfolder'))
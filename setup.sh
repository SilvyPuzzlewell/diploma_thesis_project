#!/bin/bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python load_nltk.py

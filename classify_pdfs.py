#!~/anaconda3/envs/pynco3.6-env/bin/python
# -*- coding: utf-8 -*-

import unicodecsv as csv
import os, sys, re

from datetime import datetime

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO,StringIO

import PyPDF2
import string
from collections import Counter
import nltk
from nltk.corpus import words

def extract_text_from_pdf(pdf_path):
    
    # These lines seem to be something always needed for initialization
    resource_manager = PDFResourceManager()
    fake_file_handle = StringIO()
    # Perform layout analysis for all text.  This was included
    # as sometimes text is read in with no spaces, and it forms
    # one long string.  This helps.
    laparams = LAParams()
    setattr(laparams, 'all_texts', True)
    converter = TextConverter(resource_manager, fake_file_handle,laparams=laparams)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    # end initialization

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            
        text = fake_file_handle.getvalue()
    #endwith

    # close open handles
    converter.close()
    fake_file_handle.close()
    
    if text:
        # This appears to give text with no spaces
        # Also note that some words are returned hyphenated (due to 
        # line breaks), so searching for those words will
        # turn up nothing.
        return text
    else:
        return
    #endif
#enddef

def create_dictionary(wordlist,keyword_file,rejected_words_file):
    # From a word list, ask the user if they want to add this
    # word to a list of "good" words that I will use to fingerprint
    # articles, or a list of "bad" words that will be ignored.
    # Do a first triage with words found in the WordNet database.
    # Not all words are there, but it's a good start.

    print("Working on creating dictionary.")

    # Take a list of words from some classic books from nltk.

    try:
        f_keyword=open(keyword_file,"r+")
    except:
        f_keyword=open(keyword_file,"w")
    #endtry

    try:
        f_reject=open(rejected_words_file,"r+")
    except:
        f_reject=open(rejected_words_file,"w")
    #endif

    # This is very slow, likely because words.words() is large
    for word in wordlist:
        if word.lower() in words.words():
            print("Found word! ",word)
        else:
            print("Did not find word! ",word)
        #endif
    #endfor

    f_keyword.close()
    f_reject.close()
#enddef

def create_wordlist(text):
    # Assumes an input string of text.  Find out what the ten most
    # popular words are, excluding some basic words like "the" and "a".
    # Return those words and their counts.

    # This is far from perfect.  Units, for example, show up as words.

    wordlist=text.upper()
    # join together any words seperated by a hyphen by removing the hypen and
    # any non-letter that comes after (such as a newline)
    wordlist=re.sub(r'-\W+',r'',wordlist)
    # Replace all non-letters by spaces before splitting
    wordlist=re.sub(r'\W',r' ',wordlist)
    wordlist=re.sub(r'\d',r' ',wordlist)

    # split the text into individual words based on whitespace
    wordlist=wordlist.split()

    return wordlist
#enddef

def process_wordlist(wordlist):
    # Takes a list of words.  Find out what the ten most
    # popular words are, excluding some basic words like "the" and "a".
    # Return those words and their counts.

    print("How many words in this text? {0}".format(len(wordlist)))
    wordcounts=Counter(wordlist)
    for word in wordcounts.keys():
        print(word,wordcounts[word])

    return
        
#enddef

    
if __name__ == '__main__':

    # This file contains a list of words that we will use
    # to classify according to, based on user input.  We will
    # also include words in a standard dictionary.
    keyword_file="keywords.txt"

    # This file contains a list of words that we have asked
    # the user if they are "good" and the user has responded
    # that they are not.
    rejected_words_file="rejected_words.txt"

    # This needs to be done the first time the script is run, if you
    # don't already have words
    #nltk.download('words')


    directory_in_str=os.getcwd()
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"): 
            print("Processing {}".format(filename))
            if filename == "abramowitz2012.pdf":
                
                # Get the text in a string format.
                text=extract_text_from_pdf(filename)
                print(text)

                # Create a list of words from this string.
                wordlist=create_wordlist(text)

                # Check to see if these words are good or not
                create_dictionary(wordlist,keyword_file,rejected_words_file)
               
                # Get some statistics on these words
                word_counts=process_wordlist(wordlist)

            #endif
        #endif




# Some documentation that I found
#Parameters:	
#
#    line_overlap – If two characters have more overlap than this they are considered to be on the same line. The overlap is specified relative to the minimum height of both characters.
#    char_margin – If two characters are closer together than this margin they are considered to be part of the same word. If characters are on the same line but not part of the same word, an intermediate space is inserted. The margin is specified relative to the width of the character.
#    word_margin – If two words are are closer together than this margin they are considered to be part of the same line. A space is added in between for readability. The margin is specified relative to the width of the word.
#    line_margin – If two lines are are close together they are considered to be part of the same paragraph. The margin is specified relative to the height of a line.
#    boxes_flow – Specifies how much a horizontal and vertical position of a text matters when determining the order of text boxes. The value should be within the range of -1.0 (only horizontal position matters) to +1.0 (only vertical position matters).
#    detect_vertical – If vertical text should be considered during layout analysis
#    all_texts – If layout analysis should be performed on text in figures.

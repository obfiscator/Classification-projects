############################################
# Goal: Based on a set of .pdf files, find other .pdfs which
#       satisfy that tag.
#
# Usage: python classify_pdfs.py --tag machine_learning [--files test1.pdf,test2.pdf,...]
#
#        If a tag is given, the file list is taken from a pre-existing list.
#        If a tag is not given, a file list must be specified.
#
############################################

###############
#!~/anaconda3/envs/pynco3.6-env/bin/python
# -*- coding: utf-8 -*-

import argparse
import unicodecsv as csv
import os, sys, re, traceback

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
import json
################


# Use a global debug variable to quickly shift between some test cases
# and the amount of output I write.
ldebug=True

####### Parse input arguments #########
parser = argparse.ArgumentParser(description='Classify scientific journal articles in .pdf format based on the words which appear in them.')
parser.add_argument('--tag', dest='tag', action='store',required=False,
                    choices=["machine_learning", "fluxcom"],
                   help='use pre-selected articles')
parser.add_argument('--files', dest='filelist', action='store',required=False,
                   help='a list of files to use to build the model, in format "file1.pdf,file2.pdf,file3.pdf"')

args = parser.parse_args()

print("######################### INPUT VALUES #########################")
if not args.tag and not args.filelist:
    print("At least one of the --tag or --files flags must be used!")
    traceback.print_stack(file=sys.stdout)
    sys.exit(1)
elif args.tag and args.filelist:
    print("******* WARNING ********")
    print("Since both --tag and --files flags are specified, I will ignore --files.")
    print("******* END WARNING ********")
    args.filelist=""
#endif

if args.tag and not args.filelist:
    if args.tag == "machine_learning":
        filelist="ogorman2018.pdf,tramontana2016.pdf,xu2018.pdf"
    elif args.tag == "fluxcom":
        filelist="tramontana2016.pdf,jung2020.pdf"
    else:
        print("Do not have any files for this tag.")
        print("tag: ",args.tag)
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)
    #endif
elif not args.tag and args.filelist:
    filelist=args.filelist
else:
    print("Not sure how I get here?")
    print("tag: ",args.tag)
    print("filelist: ",args.filelist)
    traceback.print_stack(file=sys.stdout)
    sys.exit(1)
#endif

files=filelist.split(",")
print("I will train a model based on the following files: ",files)

####### Define some subroutines ##########

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

def create_dictionary(wordlist):
    # Trim down a word list, since the word list will include a lot
    # of things that aren't really words.
    # Do a first triage with words found in the WordNet database.
    # Not all words are there (including plurals), but it's a good start.
    # This routine takes a little long.

    print("Working on creating dictionary.")

    # This is very slow, likely because words.words() is large
    # Is it really necessary?  I guess better to compare against
    # a set of words.  I don't want to lose the frequency information,
    # though.  So, first do that, and then put the frequency
    # information in a dictionary to keep for later.
    wordcounts=Counter(wordlist)

    # the Counter object has a nice feature that lets us grab the
    # most popular words.  The problem is that it does output a 
    # list instead of a dictionary.
    if ldebug:
        wordlist=Counter(wordlist).most_common(50)
        wordcounts={}
        for word,freq in wordlist:
            wordcounts[word]=freq
        #endfor
    #endif

    retained_words_freq={}

    # I will keep any word that I find in the dictionary.
    # Notice that I need to convert the word to lowercase, since
    # it appears that all words in words are lowercase.
    for word in wordcounts.keys():

        # Try a couple different variations of the word,
        # stripping off the trailing S and ES.  The way this
        # loop is written, I cannot distinguish between "a" and
        # "as", for example, but I hope that cases like that are
        # few and far between.  I feel this is still much better
        # than missing all plural words, or as counting plural and
        # singular words as seperate enteries.
        wordtry=[word.lower()]
        if word.endswith(("s","S")):
            wordtry.append(re.sub(r"s$",r"",word.lower()))
        #endif
        if word.endswith(("es","ES")):
            wordtry.append(re.sub(r"es$",r"",word.lower()))
        #endif
        if word.endswith(("ies","IES")):
            wordtry.append(re.sub(r"ies$",r"y",word.lower()))
        #endif

        for word_variation in wordtry:
            if ldebug:
                print("Trying word: ",word_variation)
            #endif
            if word_variation in words.words():
                if ldebug:
                    print("Found word: ",word_variation,word)
                #endif
                retained_words_freq[word_variation]=wordcounts[word]
            else:
                if ldebug:
                    print("Did not find word: ",word_variation)
                #endif
            #endif
        #endfor
    #endfor

    return(retained_words_freq)

#enddef

def create_wordlist(text):
    # Assumes an input string of text.  Find out what the ten most
    # popular words are, excluding some basic words like "the" and "a".
    # Return those words and their counts.

    # This is far from perfect.  Units, for example, show up as words.

    # Making all capital letters harmonizes things, but now acronymes
    # that are words will be counted as words.  Nothing to be done, though.
    wordlist=text.upper()

    # join together any words seperated by a hyphen by removing the hypen and
    # any non-letter that comes after (such as a newline)
    wordlist=re.sub(r'-\W+',r'',wordlist)
    # Replace all non-letters by spaces before splitting
    wordlist=re.sub(r'\W',r' ',wordlist)
    wordlist=re.sub(r'\d',r' ',wordlist)

    # Do I try to undo plurals here?  Removing s and es at the end
    # of words?  Plurals aren't recognized in the NLTK word list
    # that I check against later.  This won't catch a word like 
    # "energies".  Perhaps it's better to do this leter?
    #wordlist=re.sub(r'ES\s',r' ',wordlist)
    #wordlist=re.sub(r'S\s',r' ',wordlist)

    # split the text into individual words based on whitespace
    wordlist=wordlist.split()

    return wordlist
#enddef

####### Execute the main code ########

if __name__ == '__main__':

    # This needs to be done the first time the script is run, if you
    # don't already have words
    #nltk.download('words')


    # First, build a model with our file list
    for filename in files:
        if filename.endswith(".pdf"): 
            print("Processing {}".format(filename))
            # Check to see if our processed .txt file
            # exists.  If so, we don't want to redo all the work.
            txt_filename=re.sub(r".pdf$",r".txt",filename)
            try:
                f_txt=open(txt_filename,"r")
                f_txt.close()
                lfound=True
            except:
                lfound=False
            #endif
            print("Do we have txt file? ",txt_filename,lfound)
            # Do we have a processed .txt file already?
            if not lfound:
                #if filename == "abramowitz2012.pdf":
                if True:
                
                    # Get the text in a string format.
                    text=extract_text_from_pdf(filename)
                    if ldebug:
                        print(text)
                    #endif
                    
                    # Create a list of words from this string.
                    wordlist=create_wordlist(text)
                    
                    # Check to see if these words are good or not
                    retained_words_freq=create_dictionary(wordlist)
                    
                    # Write this to a .txt file.
                    with open(txt_filename, 'w') as file:
                        file.write(json.dumps(retained_words_freq))
                    #endwith

                #endif
            #endif
        #endif
    #endfor

    traceback.print_stack(file=sys.stdout)
    sys.exit(1)

    directory_in_str=os.getcwd()
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"): 
            print("Processing {}".format(filename))
            # Check to see if our processed .txt file
            # exists.  If so, we don't want to redo all the work.
            txt_filename=re.sub(r".pdf$",r".txt",filename)
            try:
                f_txt=open(txt_filename,"r")
                f_txt.close()
                lfound=True
            except:
                lfound=False
            #endif
            print("Do we have txt file? ",txt_filename,lfound)
            # Do we have a processed .txt file already?
            if not lfound:
                #if filename == "abramowitz2012.pdf":
                if True:
                
                    # Get the text in a string format.
                    text=extract_text_from_pdf(filename)
                    if ldebug:
                        print(text)
                    #endif
                    
                    # Create a list of words from this string.
                    wordlist=create_wordlist(text)
                    
                    # Check to see if these words are good or not
                    retained_words_freq=create_dictionary(wordlist)
                    
                    # Write this to a .txt file.
                    with open(txt_filename, 'w') as file:
                        file.write(json.dumps(retained_words_freq))
                    #endwith

                #endif
            #endif
        #endif
    #endfor
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

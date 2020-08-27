############################################
# Goal: Based on a set of .pdf files, find other .pdfs which
#       satisfy that tag.
#
# Usage: python classify_pdfs.py --tag machine_learning [--files test1.pdf,test2.pdf,...]
#
#        If a tag is given, the file list is taken from a pre-existing list.
#        If a tag is not given, a file list must be specified.
#
#        As the operation to get the word frequency can be time-consuming,
#          I write the word frequency to a file.  If this .txt file is
#          found, just use those values.  If the .txt file is not found,
#          do the frequency analysis.  There is a keyword to override this
#          feature and do the analysis every time.
#
#        One challenging point of this routine is creating the machine
#        model.  I take a word histogram from every file to create
#        a sort of "signature" of the types of articles that I am interested
#        in, using only the most common words.  These words are not the same
#        between files.  From this analysis, though, I create a list
#        of important words by combining all the words together.  This
#        gives me my master word list for which I calculate the frequency
#        for all the files of interest, and for which the model is trained.
#        Each word in this master list becomes a feature.
#
#        Once the model is trained, it goes through all .pdfs in the directory
#        and tries to predict if each .pdf is part of this group, based on the
#        word frequency count of this master list.
#
############################################

###############
#!~/anaconda3/envs/pynco3.6-env/bin/python
# -*- coding: utf-8 -*-

import argparse
import unicodecsv as csv
import os, sys, re, traceback

from sklearn import svm
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
import numpy as np
################


# Use a global debug variable to quickly shift between some test cases
# and the amount of output I write.
ldebug=True

####### Parse input arguments #########
parser = argparse.ArgumentParser(description='Classify scientific journal articles in .pdf format based on the words which appear in them.')
parser.add_argument('--tag', dest='tag', action='store',required=False,
                   help='use pre-selected articles')
parser.add_argument('--files', dest='filelist', action='store',required=False,
                   help='a list of files to use to build the model, in format "file1.pdf,file2.pdf,file3.pdf"')
parser.add_argument('--file_class', dest='fileclass', action='store',required=False,
                   help='classification of the files as part of the group we are looking for, in format "yes,yes,no"')
parser.add_argument('--nwords', dest='nwords', action='store',required=False, type=int, default=50,
                   help='The number of most common words in each article that we attempt to build a model with.')
parser.add_argument('--ignore_txt', dest='ignore_txt', action='store',required=False, default="False",
                   help='If a .txt file with word frequency exists, ignore it and recalculate the frequency.')

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
    if args.tag.lower() in ("machine_learning","ml"):
        filelist="jung2020.pdf,ogorman2018.pdf,tramontana2016.pdf,xu2018.pdf,beer2010.pdf,yousefpour2015.pdf,pingoud2006.pdf"
        fileclass="yes,yes,yes,yes,no,no,no"
    elif args.tag.lower() == "fluxcom":
        filelist="jung2020.pdf,tramontana2016.pdf,jung2020.pdf,beer2010.pdf,yousefpour2015.pdf"
        fileclass="yes,yes,yes,no,no,no"
    else:
        print("Do not have any files for this tag.")
        print("tag: ",args.tag)
        print("Please use one of the following: machine_learning ml fluxcom")
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
file_class=fileclass.split(",")

nwords_retained=args.nwords
print("I will build a model using the {} most common words from each article above.".format(nwords_retained))

possible_true_values=["true","t","yes","y"]
lignore_txt=args.ignore_txt
if lignore_txt.lower() in possible_true_values:
    lignore_txt=True
    print("Recalculating all word frequencies.")
else:
    lignore_txt=False
    print("If a .txt file exists for an article, I will take word frequencies from there.")
#endif

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
    wordlist=Counter(wordlist).most_common(nwords_retained)
    wordcounts={}
    for word,freq in wordlist:
        wordcounts[word]=freq
    #endfor

    retained_words_freq={}

    # I will keep any word that I find in the dictionary.
    # Notice that I need to convert the word to lowercase, since
    # it appears that all words in words are lowercase.
    for word in wordcounts.keys():

        # There are some words that are just not helpful to us.
        # I don't want the learning algorithm to put too much
        # weight into these, so let's just skip them.

        # Common short words
        if word.lower() in ["the", "and", "from", "for", "but", "with", "over", "was", "were", "are", "can", "not"]:
            continue
        #endif

        # Words that are only one or two letters long.  These could be 
        # variable names: x, y, z.
        if len(word) <= 2:
            continue
        #endif

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
    # Assumes an input string of text.  Breaks this up into a list of words.

    # This is far from perfect.  Units, for example, show up as words.  We
    # will do additional processing in another routine, create_fictionary.

    # Making all capital letters harmonizes things, but now acronymes
    # that are words will be counted as words.  Nothing to be done, though.
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

####### Execute the main code ########

if __name__ == '__main__':

    # This needs to be done the first time the script is run, if you
    # don't already have words
    #nltk.download('words')

    feature_words=[]
    master_file_word_hist=[]

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
            if lignore_txt:
                print("Ignoring .txt files.  Redoing frequency analysis.")
                lfound=False
            #endif

            # Do we have a processed .txt file already?
            if not lfound:
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

            else:
                # read it in from the .txt file
                # This is code from stackexchange to enable
                # reading in a json file as a Python dict.
                def js_r(filename):
                    with open(filename) as f_in:
                        return(json.load(f_in))
                    #endwith
                #enddef

                retained_words_freq=js_r(txt_filename)

            #endif

            for word,freq in retained_words_freq.items():
                if word not in feature_words:
                    feature_words.append(word)
                #endif
            #endfor

            master_file_word_hist.append(retained_words_freq)

        else:
            print("Oops!  You passed me a file that is not a .pdf file.")
            print("Please remove the following filename from the list: ",filename)
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)
        #endif
    #endfor

    nfeatures=len(feature_words)
    print("We have {} features: ".format(nfeatures),feature_words)


    # Create our training dataset.
    ntraining_files=len(files)
    training_data=np.zeros((ntraining_files,nfeatures),dtype=int)
    for ifile in range(ntraining_files):
        for word,freq in master_file_word_hist[ifile].items():
            iindex=feature_words.index(word)
            training_data[ifile,iindex]=freq
        #endfor
    #endfor

    # Now train our ML model.  Which algorithm to choose?  Microsoft
    # recommends choosing the algorithm according to what you want
    # to do, and the parameters of your request: accuracy, training time,
    # linearity, number of parameters, number of features.

    # For this case, we are trying to predict between two categories: 
    # "article of interest" or "not article of interest".  This gives several
    # possiblities: two-class support vector machine, 
    # two-class average perceptron, two-class decision forest, 
    # two-class logistic regression, two-class boosted decision tree, 
    # two-class neural network.

    # To start with, try the SVM.  It's effective when the number of
    # dimensions is greater than the number of samples, as is the
    # case for us (100 features, only a few samples).  But overfitting is
    # a danger, so we should be careful with the Kernel functions.
    clf=svm.SVC()
    clf.fit(training_data,file_class)
    print("Model trained.")

    # Now the model is trained...try it out.

    for file in os.listdir("."):
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
            if lignore_txt:
                print("Ignoring .txt files.  Redoing frequency analysis.")
                lfound=False
            #endif

            # Do we have a processed .txt file already?
            if not lfound:
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

            else:
                # read it in from the .txt file
                # This is code from stackexchange to enable
                # reading in a json file as a Python dict.
                def js_r(filename):
                    with open(filename) as f_in:
                        return(json.load(f_in))
                    #endwith
                #enddef

                retained_words_freq=js_r(txt_filename)

            #endif

            # Now I have a word frequency histogram.  I need to create
            # a vector with the same dimensions as our training data.
            test_vector=np.zeros((nfeatures),dtype=int)
            for word,freq in retained_words_freq.items():
                if word in feature_words:
                    iindex=feature_words.index(word)
                    test_vector[iindex]=freq
                #endif
            #endfor

            result=clf.predict(test_vector.reshape(1,-1))
            print(filename,result)

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

# Classification-projects
A place to put small projects related to classification problems

# Project 1
classify_pdfs.py

A first test using 18 classified .pdfs (6 "yes", 12 "no") showed poor results: the SVM model showed a recall_macro score of 0.5,0.5 for 2 cross validations, which seems equivalent to random chance.  This test extracted the 100 most common words from each file, combining all these words into a single set of features, and then populated the features without re-calculating the histograms, i.e. if word X was the 101st most common word in file Y and the 100th most common word in file Z, it was used in the score for file Z but not file Y.

Re-calculating the histogram (i.e., populating every feature for every file) for the same dataset did not improve anything.  I also tried using the 10 most frequent words and the 1000 most frequent words: same 0.5 score in cross-validation.

Current hypothesis is that the training set is too small, or I have too many features (or both).  I may try increasing to 100 classified .pdfs.  If that is too small, I do not think this is feasible for my resources.

# Project 2
classify_orchidee_timeseries.py

The rest of the files are modules called by the above two projects.

import re
import itertools
import math
import cPickle as pickle
import numpy as np
import cv2

data_dir = sys.argv[1]
stopwords_file = sys.argv[2]

STRIPCHARS = '-_\'"'
WORD_REGEX = re.compile('[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'a-zA-Z\-]+')
def split_words_from_email(email, stopwords):
    words = map(lambda w: w.lower().strip(STRIPCHARS),
                WORD_REGEX.findall(email))

    return [w for w in words if len(w) >= 3 and w not in stopwords]

# read emails and remove stop words
with open(stopwords_file) as f:
    stopwords = set(str.strip(word) for word in f)

all_words = set()
word_count_in_email = {} # (word, email) -> how many times word is in email
emails_with_word_count = {} # word -> how many emails contain the word

email_files = glob.glob(os.path.join(data_dir, '*.txt'))
email_names = map(os.path.basename, email_files)

for i, ef in enumerate(email_files):
    email = email_names[i]

    with open(ef) as f:
        sys.stderr.write('\rReading email {} of {}... '.format(i + 1, len(email_names)))

        words_in_email = split_words_from_email(f.read(), stopwords)
        unique_words = set(words_in_email)
        all_words |= unique_words

        for w in words_in_email:
            word_count_in_email[w, email] = word_count_in_email.setdefault((w, email), 0) + 1

        for w in unique_words:
            emails_with_word_count[w] = emails_with_word_count.setdefault(w, 0) + 1

print 'Done!'

# calculate tf (term frequency) and idf (inverse document frequency) for
# each word and email, and pick the words we'll use as features

idf = {}
feature_words = []
for w in all_words:
    ratio = len(email_files) / float(emails_with_word_count[w])
    idf[w] = math.log(ratio)

tf = word_count_in_email

print 'Using {} unique words.'.format(len(all_words))

# calculate the term vector for each email

term_vectors = []

out_queue = []
OUT_QUEUE_MAXSIZE = 1024
with open('term_vectors', 'w') as tvf:
    for i, email in enumerate(email_names):
        sys.stderr.write('\rCalculating tf {} of {}... '.format(i + 1, len(email_names)))

        tvec = tuple(idf[w] * tf.get((w, email), 0) for w in all_words)

        # normalize to unit length
        norm = np.linalg.norm(tvec)
        tvec = tuple(e / norm for e in tvec)

        term_vectors.append(tvec)

        serialized = repr((email, tvec))
        assert eval(serialized) == (email, tvec)

        out_queue.append(serialized)
        if len(out_queue) == OUT_QUEUE_MAXSIZE:
            while out_queue:
                tvf.write(out_queue.pop(0) + '\n')

    while out_queue:
        tvf.write(out_queue.pop(0) + '\n')

term_vectors = np.array(term_vectors, dtype = 'float32')
distortion, clusters, means = cv2.kmeans(
        term_vectors, K = 23, criteria = (cv2.TERM_CRITERIA_MAX_ITER, 1000, 0),
        attempts = 5, flags = cv2.KMEANS_RANDOM_CENTERS)

print distortion

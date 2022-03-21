import re
import nltk
import pickle


longest_word_length = 18
verbose = 0
max_edit_distance = 2


file = open('dictionary', "rb")
dictionary = pickle.load(file)
file.close()


def dameraulevenshtein(seq1, seq2):

    oneago = None
    this_row = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):

        twoago, oneago, this_row = (
            oneago, this_row, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = this_row[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            this_row[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                this_row[y] = min(this_row[y], twoago[y - 2] + 1)
    return this_row[len(seq2) - 1]


# --------------------------------------------------------------------------------------------------------------------


def get_suggestions(string, silent=False):
    """return list of suggested corrections for potentially incorrectly
       spelled word"""
    if (len(string) - longest_word_length) > max_edit_distance:
        if not silent:
            print("no items in dictionary within maximum edit distance")
        return []

    suggest_dict = {}
    min_suggest_len = float('inf')

    queue = [string]
    q_dictionary = {}

    while len(queue) > 0:
        q_item = queue[0]
        queue = queue[1:]
        # print(q_item)

        if ((verbose < 2) and (len(suggest_dict) > 0) and ((len(string) - len(q_item)) > min_suggest_len)):
            break

        # process queue item
        if (q_item in dictionary) and (q_item not in suggest_dict):
            if dictionary[q_item][1] > 0:
                # word is in dictionary, and is a word from the corpus, and not already in suggestion list so add to
                # suggestion dictionary, indexed by the word with value (frequency in corpus, edit distance)
                # q_items that are not the input string are shorter
                # than input string since only deletes are added (unless manual dictionary corrections are added)
                assert len(string) >= len(q_item)
                suggest_dict[q_item] = (
                    dictionary[q_item][1], len(string) - len(q_item))

                # early exit
                if (verbose < 2) and (len(string) == len(q_item)):
                    break
                elif (len(string) - len(q_item)) < min_suggest_len:
                    min_suggest_len = len(string) - len(q_item)

            for sc_item in dictionary[q_item][0]:
                if sc_item not in suggest_dict:

                    assert len(sc_item) > len(q_item)

                    assert len(q_item) <= len(string)

                    if len(q_item) == len(string):
                        assert q_item == string
                        item_dist = len(sc_item) - len(q_item)

                    assert sc_item != string

                    item_dist = dameraulevenshtein(sc_item, string)

                    # do not add words with greater edit distance
                    if (verbose < 2) and (item_dist > min_suggest_len):
                        pass
                    elif item_dist <= max_edit_distance:
                        # should already be in dictionary if in suggestion list
                        assert sc_item in dictionary
                        suggest_dict[sc_item] = (
                            dictionary[sc_item][1], item_dist)
                        if item_dist < min_suggest_len:
                            min_suggest_len = item_dist

                    if verbose < 2:
                        suggest_dict = {
                            k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

        assert len(string) >= len(q_item)

        # do not add words with greater edit distance
        if (verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
            pass
        elif (len(string) - len(q_item)) < max_edit_distance and len(q_item) > 1:
            for c in range(len(q_item)):  # character index
                word_minus_c = q_item[:c] + q_item[c + 1:]
                if word_minus_c not in q_dictionary:
                    queue.append(word_minus_c)
                    q_dictionary[word_minus_c] = None

    as_list = suggest_dict.items()
    outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

    if verbose == 0:
        return outlist[0]
    else:
        return outlist


def best_word(s, silent=False):
    try:
        return get_suggestions(s, silent)[0]
    except:
        return None


def spell_corrector(word_list, all_words) -> str:
    result_list = []
    for word in word_list:
        # print(word)
        if word not in all_words:
            # print(word)
            suggestion = best_word(word, silent=True)
            if suggestion is not None:
                result_list.append(suggestion)
            else:
                result_list.append(word)
        else:
            result_list.append(word)
    return " ".join(result_list)


# file1 = open('word_dictionary', "rb")
# word_dictionary = pickle.load(file1)
# file1.close()

# sample_text = 'wherr'
# tokens = nltk.word_tokenize(sample_text)
# print('original text: ' + sample_text)
# correct_text = spell_corrector(tokens, word_dictionary)
# print('corrected text: ' + correct_text)

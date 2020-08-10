#!/usr/bin/env python3
import json
import operator
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

freq_words = os.path.join(BASE_DIR, "data/word_freqs.json")
stop_list = os.path.join(BASE_DIR, "data/StopList.txt")

if not os.path.isfile(freq_words):
    raise FileNotFoundError("Unable to find '{freq_words}' file in current path")
elif not os.path.isfile(stop_list):
    raise FileNotFoundError("Unable to find '{stop_list}' file in current path")


class Rake(object):
    """Rapid automated keyword extraction implementation

    Try and store the rake object globally as it will help speed up exection times during
    initialization

    Params:
        phrase_length (int): Maximum phrase length to collect
        min_word_size (int): Minimum length of word allowed to be a phrase

    Attributes:
        stop_words_pattern (re.Pattern): pattern to match all stop words in input text
        frequent_words (dict): large english corpus with frequencies from BYU
    """

    def __init__(self, phrase_length=2, min_word_size=3):
        self.stop_list = self._load_stop_words()
        self.stop_words_pattern = self._build_stop_word_regex()
        self.frequent_words = self._get_frequent_english_words()
        self.phrase_length = phrase_length
        self.min_word_size = min_word_size

    def _is_number(self, s):
        """Determins if input string is a number"""
        s = s.replace("%", "")
        return str.isdigit(s)

    def _load_stop_words(self):
        """Load stop words from StopList file"""
        with open(stop_list, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    def _build_stop_word_regex(self):
        """Creates a pattern to match stop words from the stop list"""
        stop_word_regex_list = []
        for word in self.stop_list:
            word_regex = r"\b" + word + r"(?![\w-])"  # added look ahead for hyphen
            stop_word_regex_list.append(word_regex)
        stop_word_pattern = re.compile("|".join(stop_word_regex_list), re.IGNORECASE)
        return stop_word_pattern

    def _get_frequent_english_words(self):
        """Returns a dictionary of frequent english words from BYU corpus"""
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), freq_words), "r"
        ) as f:
            words = json.load(f)
        return words

    def _split_sentences(self, text):
        """Utility function to return a list of sentences."""
        sentence_delimiters = re.compile(
            "[.!?,;:\t\\\\\"\\(\\)\\'\u2019\u2013]|\\s\\-\\s"
        )
        sentences = sentence_delimiters.split(text)
        return sentences

    def _separate_words(self, text):
        """Separates words based on length and if phrase contains numbers"""
        splitter = re.compile("[^a-zA-Z0-9_\\+\\-/]")
        words = []
        for single_word in splitter.split(text):
            current_word = single_word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if (
                len(current_word) > self.min_word_size
                and current_word != ""
                and not self._is_number(current_word)
            ):
                words.append(current_word)
        return words

    def _generate_candidate_keywords(self, sentence_list, stopword_pattern):
        """Returns list of phrases that matches stopword regex"""
        phrase_list = []
        for s in sentence_list:
            tmp = re.sub(stopword_pattern, "|", s.strip())
            phrases = tmp.split("|")
            for phrase in phrases:
                phrase = phrase.strip().lower()
                if phrase != "":
                    phrase = re.sub(
                        "[\"'\â€œ\â€]+[\S+\n\r\s]+", "", phrase
                    )  # remove prepended punctuation and spaces
                    phrase_list.append(phrase)
        return phrase_list

    def _get_idf_score(self, word):
        """Based off of frequency in English language"""
        if word in self.frequent_words:
            rank = self.frequent_words[word]["rank"]
            idf_rank = (float(rank + 5000) ** (-0.8)) * 1000
            return idf_rank
        else:
            return 0

    def _calculate_word_scores(self, phrase_list):
        """Calculates word scores based on frequencies and degree"""
        phrase_list_length = len(phrase_list)
        word_frequency = {}
        word_degree = {}

        for phrase in phrase_list:
            word_list = self._separate_words(phrase)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1

            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree

        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]

        word_score = {}
        for word, freq in word_frequency.items():
            frequency = 100 * float(freq / phrase_list_length)
            idf = self._get_idf_score(word)
            score = frequency - (
                frequency * idf
            )  # reduce score for frequent words by IDF value

            word_score.setdefault(word, 0)
            word_score[word] = score

        return word_score

    def _generate_candidate_keyword_scores(self, phrase_list, word_score):
        """Calculates phrase scores"""
        keyword_candidates = {}
        for phrase in phrase_list:
            phrase_words = self._separate_words(phrase)
            length = len(phrase.split(" "))  # length of separation of phrase by spaces

            if not phrase_words:
                continue
            elif length > self.phrase_length:
                continue
            elif len([i for i in phrase_words if self._is_number(i)]) > 0:
                continue

            candidate_score = 0

            for word in phrase_words:
                candidate_score += word_score[word]

            # print(phrase_words, length)

            if candidate_score > 0:
                if length > 1:
                    candidate_score = float(candidate_score / (length * 1.05))
                keyword_candidates[phrase] = candidate_score
        return keyword_candidates

    def get_phrases(self, text, length=None):
        """Returns a sorted list of phrases"""
        sentence_list = self._split_sentences(text)

        phrase_list = self._generate_candidate_keywords(
            sentence_list, self.stop_words_pattern
        )

        word_scores = self._calculate_word_scores(phrase_list)

        keyword_candidates = self._generate_candidate_keyword_scores(
            phrase_list, word_scores
        )

        sorted_keywords = sorted(
            keyword_candidates.items(), key=operator.itemgetter(1), reverse=True
        )

        if length:
            if length > len(sorted_keywords):
                return sorted_keywords  # return all words if length is too high
        else:
            length = 5 + int(
                (len(sorted_keywords) / 10)
            )  # calculate 'suggested' length

        return sorted_keywords[:length]
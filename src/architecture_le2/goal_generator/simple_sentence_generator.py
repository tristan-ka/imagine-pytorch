import numpy as np
from nltk import word_tokenize


class SentenceGenerator:
    def __init__(self):
        self.sentences_set = set()
        self.max_len = 0
        pass

    def update_model(self, sentences_list):
        new_sentence_set = set(sentences_list).difference(self.sentences_set)
        self.sentences_set = self.sentences_set.union(sentences_list)
        new_sentence_tokenized = [word_tokenize(s) for s in new_sentence_set]
        for new_sentence in new_sentence_tokenized:
            if len(new_sentence) > self.max_len:
                self.max_len = len(new_sentence)
        for new_sentence in new_sentence_tokenized:
            if len(new_sentence) < self.max_len:
                new_sentence += [' ' for _ in range(self.max_len - len(new_sentence))]
        return new_sentence_tokenized

    def generate_sentences(self, n=1):
        pass

class SentenceGeneratorHeuristic(SentenceGenerator):
    def __init__(self, sentences=None):
        super().__init__()
        self.sentence_types = []
        self.word_equivalence = []
        if sentences is not None:
            self.update_model(sentences)


    def update_model(self, sentences_list):
        new_sentences_tokenized = super().update_model(sentences_list)
        # pad type sentences
        for type_sent in self.sentence_types:
            if len(type_sent) < self.max_len:
                type_sent += [' ' for _ in range(self.max_len - len(type_sent))]

        if len(self.sentence_types) == 0:
            self.sentence_types.append(new_sentences_tokenized[0])
            new_sentences_tokenized = new_sentences_tokenized[1:]
        # for each sentence type, compute semantic distances with every new sentences tokenized
        for s_new in new_sentences_tokenized:
            match = False
            for index, s_type in enumerate(self.sentence_types):
                max_len = max(len(s_type), len(s_new))
                min_len = min(len(s_type), len(s_new))
                confusion = np.array([w1 == w2 for w1, w2 in zip(s_type, s_new)] + [False] * (max_len - min_len))
                semantic_dist = np.sum(~confusion)
                if semantic_dist == 1:
                    match = True
                    ind_eq = int(np.argwhere(~confusion))
                    equivalent_words = (s_type[ind_eq], s_new[ind_eq])
                    if ' ' not in equivalent_words:
                        set_match = False
                        for eq_set in self.word_equivalence:
                            if not set_match:
                                if equivalent_words[0] in eq_set and equivalent_words[1] in eq_set:
                                    set_match = True
                                else:
                                    if equivalent_words[0] in eq_set:
                                        set_match = True
                                        eq_set.add(equivalent_words[1])
                                    elif equivalent_words[1] in eq_set:
                                        set_match = True
                                        eq_set.add(equivalent_words[0])

                        if not set_match:
                            self.word_equivalence.append(set(equivalent_words))
                elif semantic_dist == 0:
                    match = True
            if not match:
                self.sentence_types.append(s_new)

        # remove sets in double,
        # merge set with equivalence
        ind_remove = []
        for i in range(len(self.word_equivalence)):
            if i not in ind_remove:
                for j in range(i + 1, len(self.word_equivalence)):
                    if self.word_equivalence[i] == self.word_equivalence[j]:
                        ind_remove.append(j)

        word_equivalence_new = []
        for j in range(len(self.word_equivalence)):
            if j not in ind_remove:
                not_new = False
                for w_eq in word_equivalence_new:
                    for w in self.word_equivalence[j]:
                        if w in w_eq:
                            w_eq.union(self.word_equivalence[j])
                            not_new = True
                            break
                if not not_new:
                    word_equivalence_new.append(self.word_equivalence[j])
        self.word_equivalence = word_equivalence_new


    def generate_sentences(self, n=1):
        new_sentences = set()
        for sent_type in self.sentence_types:
            for i, word in enumerate(sent_type):
                for word_eqs in self.word_equivalence:
                    if word in word_eqs:
                        for eq in word_eqs:
                            if eq != word:
                                new_sent = sent_type.copy()
                                new_sent[i] = eq
                                while ' ' in new_sent:
                                    new_sent.remove(' ')
                                if np.unique(new_sent).size == len(new_sent):
                                    new_sent = ' '.join(new_sent.copy())
                                    if new_sent not in self.sentences_set:
                                        new_sentences.add(new_sent)
        return list(new_sentences)





if __name__ == '__main__':
    from src.architecture_le2.goal_generator.descriptions import get_descriptions

    train_descriptions, test_descriptions, _ = get_descriptions(env='big')
    generator = SentenceGeneratorHeuristic(None)
    generator.update_model(train_descriptions)
    new_descriptions = generator.generate_sentences()

    print(new_descriptions)
    print(len(new_descriptions))
    p_found_in_test = sum([d in test_descriptions for d in new_descriptions]) / len(test_descriptions)
    p_not_in_test = sum([d not in test_descriptions for d in new_descriptions]) / len(new_descriptions)
    p_in_test = sum([d in test_descriptions for d in new_descriptions]) / len(new_descriptions)
    print('Percentage of the test set found:', p_found_in_test)
    print('Percentage of the new descriptions that are not in the test', p_not_in_test)
    print('Percentage of the new descriptions that are in the test set', p_in_test)

    print('\n Sentences in the generated set, not in the test: \n', set(new_descriptions) - set(test_descriptions))
    print('\n Sentences in the test set, found by generation: \n', set(new_descriptions).intersection(set(test_descriptions)))
    print('\n Sentences in the test set, not in the generated set: \n', set(test_descriptions) - set(new_descriptions))

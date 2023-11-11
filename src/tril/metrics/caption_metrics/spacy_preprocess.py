import spacy

"""
Code adapted from: https://github.com/INK-USC/CommonGen/tree/master/evaluation/Traditional/eval_metrics # noqa
"""


class SpacyPreprocess:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")
        # keep only tagger
        for pipe in ["tok2vec", "parser", "ner", "attribute_ruler", "lemmatizer"]:
            self._nlp.remove_pipe(pipe)

    def tokenize(self, dict):
        for key in dict:
            new_sentence_list = []
            for sentence in dict[key]:
                a = ""
                for token in self._nlp(str(sentence)):
                    a += token.text
                    a += " "
                new_sentence_list.append(a.rstrip())
            dict[key] = new_sentence_list

        return dict

    def compute_preprocess(self, gts, res):
        # tokenize
        gts = self.tokenize(gts)
        res = self.tokenize(res)

        assert gts.keys() == res.keys()

        return {"gts": gts, "res": res}

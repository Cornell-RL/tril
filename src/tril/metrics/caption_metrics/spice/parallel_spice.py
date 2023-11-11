# flake8: noqa
"""
Adapted from https://github.com/INK-USC/CommonGen/tree/master/evaluation/Traditional/eval_metrics/spice
"""

from __future__ import division

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Boiler plate stuff to start the module
import jpype
import jpype.imports
import numpy as np
import spacy
from jpype.types import *


def init_java():
    # Set JAVA_HOME
    path = Path(os.path.dirname(sys.executable))
    JAVA_HOME = os.path.join(path.parent.absolute(), "lib/server/libjvm.so")
    assert os.path.exists(
        JAVA_HOME
    ), "Conda JAVA_HOME does not exist! run `conda install -c 'conda-forge/label/gcc7' openjdk` "

    # Launch the JVM
    args = (
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.math=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.text=ALL-UNNAMED",
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    jpype.startJVM(
        JAVA_HOME,
        *args,
        classpath=[f"{dir_path}/parallel_spice-1.0.jar", f"{dir_path}/lib/*"],
    )
    from edu.anu.spice import SpiceArguments, SpiceScorer
    from java.lang import Integer, Object, String
    from java.util import ArrayList

    # Assumes spice.jar is in the same directory as spice.py.  Change as needed.
    SPICE_JAR = "parallel_spice-1.0.jar"
    TEMP_DIR = "tmp"
    # CACHE_DIR = 'cache'

    # Following HuggingFace Caching Scheme
    # https://github.com/huggingface/transformers/blob/28f26c107b4a1c5c7e32ed4d9575622da0627a40/src/transformers/utils/hub.py#L74

    # GTS caching is reference caching
    tril_cache_home = os.path.expanduser(
        os.getenv(
            "TRIL_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "tril")
        )
    )

    default_cache_path = os.path.join(tril_cache_home, "metrics")
    if not os.path.exists(default_cache_path):
        os.makedirs(default_cache_path)

    # RES caching is model generation caching
    RES_SPICE_CACHE = "res_cache"


class ParallelSpice:
    """
    Main Class to compute the SPICE metric
    :param datapool: The name of the datapool being ran, for example common_gen
    :param role: Either 'metric' or 'reward'
    """

    def __init__(self, datapool: str, role: str) -> None:
        init_java()

        self._nlp = spacy.load("en_core_web_sm")
        # keep only tagger
        for pipe in ["tok2vec", "parser", "ner", "attribute_ruler", "lemmatizer"]:
            self._nlp.remove_pipe(pipe)

        assert role in ["metric", "reward"]
        self.role = role

        # Creates a temporary folder for each parall spice instance
        # su = shortuuid.ShortUUID()
        # gts_dir = os.path.join(default_cache_path, f'{datapool}_{role}_{su.random(length=8)}')
        prefix = f"{datapool}_{role}_"
        self.GTS_SPICE_CACHE = tempfile.TemporaryDirectory(
            dir=default_cache_path, prefix=prefix
        )
        self.spice = SpiceScorer()

    def float_convert(self, obj):
        try:
            return float(obj)
        except:
            return np.nan

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

    def compute_score(self, gts, res, spacy_preprocess=False):
        # tokenize
        if not spacy_preprocess:
            gts = self.tokenize(gts)
            res = self.tokenize(res)

        assert sorted(gts.keys()) == sorted(res.keys())
        imgIds = sorted(gts.keys())

        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) >= 1

            input_data.append({"image_id": id, "test": hypo[0], "refs": ref})

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # in_file = tempfile.NamedTemporaryFile(
        #    mode="w", delete=False, dir=temp_dir)
        # json.dump(input_data, in_file, indent=2)
        # in_file.close()

        # Start job
        # out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        # out_file.close()
        temp_dir = tempfile.TemporaryDirectory()
        # temp_dir = "test_cache"

        # cache_dir = os.path.join(cwd, temp_dir.name, CACHE_DIR)
        res_cache_dir = os.path.join(cwd, temp_dir.name, RES_SPICE_CACHE)
        # res_cache_dir = os.path.join(cwd, temp_dir, RES_SPICE_CACHE)
        gts_cache_dir = self.GTS_SPICE_CACHE.name

        # if not os.path.exists(cache_dir):
        #    os.makedirs(cache_dir)
        if not os.path.exists(res_cache_dir):
            os.makedirs(res_cache_dir)
        if not os.path.exists(gts_cache_dir):
            os.makedirs(gts_cache_dir)

        # // Parse test and refs from input file
        # ArrayList<Object> image_ids = new ArrayList<Object>();
        # ArrayList<String> testCaptions = new ArrayList<String>();
        # ArrayList<String> refCaptions = new ArrayList<String>();
        # ArrayList<Integer> refChunks = new ArrayList<Integer>();
        # JSONParser json = new JSONParser();
        # JSONArray input;
        image_ids = ArrayList()
        testCaptions = ArrayList()
        refCaptions = ArrayList()
        refChunks = ArrayList()

        # for (Object o : input) {
        #    JSONObject item = (JSONObject) o;
        #    image_ids.add(item.get("image_id"));
        #    testCaptions.add((String) item.get("test"));
        #    JSONArray refs = (JSONArray) item.get("refs");
        #    refChunks.add(refs.size());
        #    for (Object ref : refs){
        #        refCaptions.add((String) ref);
        #    }
        # }

        for data in input_data:
            image_ids.add(data["image_id"])
            testCaptions.add(String(data["test"]))
            refs = data["refs"]
            refChunks.add(Integer((len(refs))))
            for ref in refs:
                refCaptions.add(String(ref))

        results = self.spice.scoreBatch(
            [
                "",
                "-res_cache",
                res_cache_dir,
                "-gts_cache",
                gts_cache_dir,
                "-out",
                "",
                "-subset",
                "-silent",
            ],
            image_ids,
            testCaptions,
            refCaptions,
            refChunks,
        )
        results = json.loads(str(results))

        # Read and process results
        # with open(out_file.name) as data_file:
        #    results = json.load(data_file)
        # os.remove(in_file.name)
        # os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        individual_scores = {}
        for item in results:
            imgId_to_scores[item["image_id"]] = item["scores"]
            spice_scores.append(self.float_convert(item["scores"]["All"]["f"]))
            individual_scores[item["image_id"]] = self.float_convert(
                item["scores"]["All"]["f"]
            )
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category, score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {
                    k: self.float_convert(v) for k, v in score_tuple.items()
                }
            scores.append(score_set)

        # temp_dir.cleanup()
        return average_score, individual_scores

    def method(self):
        return "SPICE"


if __name__ == "__main__":
    # gts = {"cat#dog#boy": ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
    #       "apple#tree#boy": ["A boy is picking apples from trees."]}
    # res = {"cat#dog#boy": ["The dog is the boy's cat."],
    #       "apple#tree#boy": ["A boy is picking apples from trees and put them into bags."]}

    start = time.perf_counter()

    new_gts = {}
    new_res = {}
    for idx in range(100):
        new_gts[f"{idx}-cat#dog#boy"] = [
            f"{idx}-The dog is the boy's cat.",
            f"{idx}-The dog eats the cat of the boy.",
        ]
        new_res[f"{idx}-cat#dog#boy"] = [f"{idx}-The dog is the boy's cat."]

    metric = ParallelSpice("test", "reward")
    metric.compute_score(new_gts, new_res)

    diff = time.perf_counter() - start
    print(f"time: {diff/60:.2f} mins")

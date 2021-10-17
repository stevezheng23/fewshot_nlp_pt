# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fewshot processors and helpers """

import os
import csv
import json
import dataclasses
from collections import defaultdict
from enum import Enum
from typing import List, Optional, Union
from datasets import Dataset

from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from .utils import DataProcessor, InputExample, InputFeatures


logger = logging.get_logger(__name__)


class FewshotProcessor:
    """Base class for data processor for Fewshot datasets."""
   
    def get_support_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the support set."""
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    @classmethod
    def get_labels(cls, examples):
        """Gets the list of labels for this data set."""
        labels = set([d["label"] for d in examples])
        return sorted(list(labels))

    @classmethod
    def to_datasets(cls, examples):
        """Converts a collection of :class:`InputExample` to a `Datasets`."""
        example_dict = defaultdict(list)
        for d in examples:
            for k, v in dataclasses.asdict(d).items():
                if not v:
                    continue
                example_dict[k].append(v)
        return Dataset.from_dict(example_dict)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_json(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f.readlines()]


class Clinc150Processor(FewshotProcessor):
    """Processor for the CLINC150 dataset (Fewshot version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_support_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "support")

    def get_train_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "train")

    def get_dev_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "dev")

    def get_test_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "test")

    def get_labels(self, examples):
        """See base class."""
        labels = set([d["label"] for d in examples])
        labels = sorted(list(labels))
        return labels
    
    def to_dataset(self, examples):
        """See base class."""
        example_dict = defaultdict(list)
        for d in examples:
            for k, v in d.items():
                example_dict[k].append(v)
        return Dataset.from_dict(example_dict)

    def _create_examples(self, lines, set_type):
        """Creates examples for the supprt, train, dev and test sets."""
        if not lines:
            return []

        examples = []
        if set_type == "train" and "label" in lines[0]:
            example_dict = defaultdict(list)
            for d in lines:
                example_dict[d["label"]].append(d)
            idx = 0
            for label in example_dict:
                t_list = example_dict[label]
                n = len(t_list)
                for i in range(n):
                    for j in range(i+1,n):
                        guid = f"clinc150-{set_type}-{idx}"
                        text_a = t_list[i]["text"]
                        text_b = t_list[j]["text"]
                        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        idx += 1
        elif set_type == "train" and "label" not in lines[0]:
            for (idx, d) in enumerate(lines):
                guid = d['id'] if "id" in d else f"clinc150-{set_type}-{idx}"
                text_a = d["text1"]
                text_b = d["text2"]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=guid))
        else:
            for (idx, d) in enumerate(lines):
                guid = d['id'] if "id" in d else f"clinc150-{set_type}-{idx}"
                text_a = d["text"]
                label = d["label"] if set_type != "test" else None
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        
        return examples


fewshot_processors = {
    "clinc150": Clinc150Processor,
}

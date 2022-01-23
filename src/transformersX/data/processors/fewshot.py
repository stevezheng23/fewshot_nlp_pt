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
import random
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
   
    def get_support_examples(self, data_file):
        """Gets a collection of :class:`InputExample` for the support set."""
        raise NotImplementedError()

    def get_train_examples(self, data_file, per_label_limit=None):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_file):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_file):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    @classmethod
    def get_labels_from_examples(cls, examples):
        """Gets the list of labels for a data set."""
        labels = set([d["label"] for d in examples])
        return sorted(list(labels))

    @classmethod
    def get_labels_from_file(cls, label_file):
        """Gets the list of labels for a label file."""
        labels = cls._read_tsv(label_file)
        return labels

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


class DefaultProcessor(FewshotProcessor):
    """Processor for the default dataset (Fewshot version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = "default"

    def get_support_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "support")

    def get_train_examples(self, data_file, per_label_limit=None):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "train", per_label_limit)

    def get_dev_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "dev")

    def get_test_examples(self, data_file):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_file)), "test")

    def _create_examples(self, lines, set_type, per_label_limit=None):
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
                p_list = []
                t_list = example_dict[label]
                n = len(t_list)
                for i in range(n):
                    for j in range(i+1,n):
                        guid = f"{self.task_name}-{set_type}-{idx}"
                        text_a = t_list[i]["text"]
                        text_b = t_list[j]["text"]
                        task = t_list[i]["task"] if "task" in t_list[i] else "default"
                        p_list.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, task=task))
                        idx += 1
                random.shuffle(p_list)
                if per_label_limit is not None and per_label_limit > 0:
                    p_list = p_list[:per_label_limit]
                examples.extend(p_list)
        elif set_type == "train" and "label" not in lines[0]:
            for (idx, d) in enumerate(lines):
                guid = d['id'] if "id" in d else f"{self.task_name}-{set_type}-{idx}"
                text_a = d["text1"]
                text_b = d["text2"]
                label = guid
                task = d["task"] if "task" in d else "default"
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=guid, task=task))
        else:
            for (idx, d) in enumerate(lines):
                guid = d['id'] if "id" in d else f"{self.task_name}-{set_type}-{idx}"
                text_a = d["text"]
                label = d["label"]
                task = d["task"] if "task" in d else "default"
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task=task))
        
        return examples


class Atis2Processor(DefaultProcessor):
    """Processor for the ATIS-2 dataset (Fewshot version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = "atis2"


class SnipsProcessor(DefaultProcessor):
    """Processor for the SNIPS dataset (Fewshot version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = "snips"


class Clinc150Processor(DefaultProcessor):
    """Processor for the Clinc150 dataset (Fewshot version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = "clinc150"


fewshot_processors = {
    "default": DefaultProcessor,
    "atis2": Atis2Processor,
    "snips": SnipsProcessor,
    "clinc150": Clinc150Processor,
}

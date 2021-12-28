# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import logging
from dataclasses import dataclass, field

from .file_utils import add_start_docstrings
from .training_args import TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class FewshotTrainingArguments(TrainingArguments):
    """
    Parameters:
        predict_strategy (:obj:`str`, `optional`, defaults to `max`):
            The predict strategy for fewshot training (e.g., `max`, `knn`, etc.).
        top_k (:obj:`int`, `optional`, defaults to 1):
            Use top K neighbors to predict label for test example.
    """

    predict_strategy: str = field(
        default="max",
        metadata={"help": "The predict strategy for fewshot training (e.g., `max`, `knn`, etc.)."},
    )    
    top_k: int = field(
        default=1,
        metadata={"help": "Use top K neighbors to predict label for test example."},
    )

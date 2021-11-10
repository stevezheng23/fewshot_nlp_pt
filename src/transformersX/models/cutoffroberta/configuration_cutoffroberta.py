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
""" CutoffRoBERTa configuration """
from collections import OrderedDict
from typing import Mapping

from ...onnx import OnnxConfig
from ...utils import logging
from ..cutoffbert.configuration_cutoffbert import CutoffBertConfig


logger = logging.get_logger(__name__)

CUTOFFROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
}


class CutoffRobertaConfig(CutoffBertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CutoffRobertaModel`.
    It is used to instantiate a CutoffRoBERTa model according to the specified arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.CutoffRobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import CutoffRobertaConfig, CutoffRobertaModel

        >>> # Initializing a CutoffRoBERTa configuration
        >>> configuration = CutoffRobertaConfig()

        >>> # Initializing a model from the configuration
        >>> model = CutoffRobertaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "cutoffroberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        cls_token_id=0,
        sep_token_id=2,
        mask_token_id=50264,
        **kwargs
    ):
        """Constructs CutoffRobertaConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
            **kwargs
        )

# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_lorabert": ["LORABERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LoraBertConfig"],
    "tokenization_lorabert": ["LoraBertTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_lorabert"] = [
        "LORABERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LoraBertForSequenceClassification",
        "LoraBertForDualPassageEncoder",
        "LoraBertModel",
        "LoraBertPreTrainedModel",
        "load_tf_weights_in_lorabert",
    ]

if TYPE_CHECKING:
    from .configuration_lorabert import LORABERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LoraBertConfig
    from .tokenization_lorabert import LoraBertTokenizer

    if is_torch_available():
        from .modeling_lorabert import (
            LORABERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LoraBertForSequenceClassification,
            LoraBertForDualPassageEncoder,
            LoraBertModel,
            LoraBertPreTrainedModel,
            load_tf_weights_in_lorabert,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)

import numpy as np

from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_DUAL_PASSAGE_ENCODER_MAPPING


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return all prediction scores or just the one of the predicted class.
    """,
)
class DualPassageEncoderPipeline(Pipeline):
    """
    Dual passage encoder pipeline using any :obj:`ModelForDualPassageEncoder`.

    This dual passage encoder pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"dual-encoder"`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_model_type(MODEL_FOR_DUAL_PASSAGE_ENCODER_MAPPING)

    def __call__(self, *args, **kwargs):
        """
        Generate pooled embeddings from the text inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`): One or several texts (or one list of texts) to get the pooled embedding of.

        Return:
            A list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **seq_embed** (list of :obj:`float`) -- The sequence embedding.
        """
        results = super().__call__(*args, **kwargs)
        return [{"seq_embed": item} for item in results]

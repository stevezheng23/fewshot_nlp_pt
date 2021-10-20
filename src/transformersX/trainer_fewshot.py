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
"""
The FewshotTrainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new fewshot task.
"""

import collections
import math
import sys
import time
import numpy as np
from logging import StreamHandler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption
from .deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from .file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, unwrap_model
from .optimization import Adafactor, AdamW, get_scheduler
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer import Trainer
from .trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from .training_args import ParallelMode, TrainingArguments
from .utils import logging

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))


logger = logging.get_logger(__name__)



class FewshotPrediction(NamedTuple):
    """
    Fewshot prediction output (always contains labels), to be used to compute metrics.

    Parameters:
        support_preds (:obj:`np.ndarray`): Predictions of the model.
        support_labels (:obj:`np.ndarray`): Targets to be matched.
        eval_preds (:obj:`np.ndarray`): Predictions of the model.
        eval_labels (:obj:`np.ndarray`): Targets to be matched.
    """

    support_preds: Union[np.ndarray, Tuple[np.ndarray]]
    support_labels: np.ndarray
    eval_preds: Union[np.ndarray, Tuple[np.ndarray]]
    eval_labels: np.ndarray


class FewshotTrainer(Trainer):
    """
    FewshotTrainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

            .. note::

                :class:`~transformers.Trainer` is optimized to work with the :class:`~transformers.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the 🤗 Transformers models.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or :obj:`eval_dataset`.
            Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is provided, an instance of
            :func:`~transformers.DataCollatorWithPadding` otherwise.
        support_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to support evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset` or :obj:`torch.utils.data.dataset.IterableDataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.

            Note that if it's a :obj:`torch.utils.data.dataset.IterableDataset` with some randomization and you are
            training in a distributed fashion, your iterable dataset should either use a internal attribute
            :obj:`generator` that is a :obj:`torch.Generator` for the randomization that must be identical on all
            processes (and the Trainer will manually set the seed of this :obj:`generator` at each epoch) or have a
            :obj:`set_epoch()` method that internally sets the seed of the RNGs used.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be
            able to choose different architectures according to hyper parameters (such as layer count, sizes of inner
            layers, dropout probabilities etc).
        compute_metrics (:obj:`Callable[[FewshotPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.FewshotPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~transformers.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a
          :class:`~transformers.PreTrainedModel` subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under ``DeepSpeed``,
          the inner model is wrapped in ``DeepSpeed`` and then again in ``torch.nn.DistributedDataParallel``. If the
          inner model hasn't been wrapped, then ``self.model_wrapped`` is the same as ``self.model``.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to :obj:`False` if model parallel or deepspeed is used, or if the default
          ``TrainingArguments.place_model_on_device`` is overridden to return :obj:`False` .
        - **is_in_train** -- Whether or not a model is currently running ``train`` (e.g. when ``evaluate`` is called
          while in ``train``)

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        support_dataset: Optional[Dataset] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_predicts: Optional[Callable[[FewshotPrediction], EvalPrediction]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.support_dataset = support_dataset
        self.compute_predicts = compute_predicts

    def _get_support_sampler(self, support_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        # Deprecated code
        if self.args.use_legacy_prediction_loop:
            if is_torch_tpu_available():
                return SequentialDistributedSampler(
                    support_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            elif is_sagemaker_mp_enabled():
                return SequentialDistributedSampler(
                    support_dataset,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            elif self.args.local_rank != -1:
                return SequentialDistributedSampler(support_dataset)
            else:
                return SequentialSampler(support_dataset)

        if self.args.world_size <= 1:
            return SequentialSampler(support_dataset)
        else:
            return ShardSampler(
                support_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )

    def get_support_dataloader(self, support_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if support_dataset is None and self.support_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        support_dataset = support_dataset if support_dataset is not None else self.support_dataset

        if is_datasets_available() and isinstance(support_dataset, datasets.Dataset):
            support_dataset = self._remove_unused_columns(support_dataset, description="support")

        if isinstance(support_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                support_dataset = IterableDatasetShard(
                    support_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                support_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        support_sampler = self._get_support_sampler(support_dataset)

        return DataLoader(
            support_dataset,
            sampler=support_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        support_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            support_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.support_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        support_dataloader = self.get_support_dataloader(support_dataset)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        output = self.evaluation_loop(
            support_dataloader,
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        support_dataset: Dataset,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            support_dataset (:obj:`Dataset`):
                Dataset to support the predictions. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        support_dataloader = self.get_support_dataloader(support_dataset)
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            support_dataloader,
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        support_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        description: str,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        preds_list = []
        labels_list = []
        dataloaders = [support_dataloader, eval_dataloader]
        for i, dataloader in enumerate(dataloaders):
            batch_size = dataloader.batch_size

            logger.info(f"***** Running - Step {i} - {description} *****")
            if isinstance(dataloader.dataset, collections.abc.Sized):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            model.eval()

            self.callback_handler.eval_dataloader = dataloader
            # Do this before wrapping.

            if is_torch_tpu_available():
                dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

            if self.args.past_index >= 0:
                self._past = None

            # Initialize containers
            # preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            preds_host = None
            labels_host = None
            # preds/labels on CPU (final containers)
            all_preds = None
            all_labels = None
            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size

                # Prediction step
                logits, labels = self.prediction_step(model, inputs, ignore_keys=ignore_keys)

                # Update containers on host
                if logits is not None:
                    logits = self._pad_across_processes(logits)
                    logits = self._nested_gather(logits)
                    preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                if labels is not None:
                    labels = self._pad_across_processes(labels)
                    labels = self._nested_gather(labels)
                    labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
                self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                    if preds_host is not None:
                        logits = nested_numpify(preds_host)
                        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    if labels_host is not None:
                        labels = nested_numpify(labels_host)
                        all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                    # Set back to None to begin a new accumulation
                    preds_host, labels_host = None, None, None

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

            # Gather all remaining tensors and put them back on the CPU
            if preds_host is not None:
                preds = nested_numpify(preds_host)
                all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
            if labels_host is not None:
                labels = nested_numpify(labels_host)
                all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

            # Number of samples
            if not isinstance(dataloader.dataset, IterableDataset):
                num_samples = len(dataloader.dataset)
            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
            # methods. Therefore we need to make sure it also has the attribute.
            elif isinstance(dataloader.dataset, IterableDatasetShard) and hasattr(dataloader.dataset, "num_examples"):
                num_samples = dataloader.dataset.num_examples
            else:
                num_samples = observed_num_examples

            # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
            # samplers has been rounded to a multiple of batch_size, so we truncate.
            if all_preds is not None:
                all_preds = nested_truncate(all_preds, num_samples)
            if all_labels is not None:
                all_labels = nested_truncate(all_labels, num_samples)
            
            preds_list.append(all_preds)
            labels_list.append(all_labels)

        support_preds, eval_preds = preds_list
        support_labels, eval_labels = labels_list

        # Metrics!
        if (
            self.compute_predicts is not None and self.compute_metrics is not None and
            support_preds is not None and support_labels is not None and
            eval_preds is not None and eval_labels is not None
        ):
            predicts = self.compute_predicts(FewshotPrediction(
                support_preds=support_preds,
                support_labels=support_labels,
                eval_preds=eval_preds,
                eval_labels=eval_labels
            ))
            predictions = predicts.predictions
            label_ids = predicts.label_ids
            metrics = self.compute_metrics(predicts)
        else:
            predictions = None
            label_ids = None
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=predictions, label_ids=label_ids, metrics=metrics, num_samples=observed_num_examples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if isinstance(raw_outputs, dict):
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                else:
                    logits_mb = raw_outputs
                logits = smp_nested_concat(logits_mb)
            else:
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (logits, labels)

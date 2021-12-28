#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for dual passage encoder on Fewshot tasks."""
# You can also adapt this script on your own text-fewshot task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric, DatasetDict

import numpy as np

import transformersX
from transformersX import (
    AutoConfig,
    AutoModelForDualPassageEncoder,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    FewshotTrainer,
    FewshotPrediction,
    EvalPrediction,
    FewshotTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformersX.trainer_utils import get_last_checkpoint
from transformersX.utils import check_min_version
from transformersX.utils.versions import require_version
from transformersX import FewshotProcessor
from transformersX import fewshot_processors as processors

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-fewshot/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples_per_label: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate the number of training examples per label to this value if set."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    support_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the support data."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the train data."}
    )
    dev_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the dev data."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the test data."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformersX/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FewshotTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.task_name is None:
        raise ValueError("Task name is required")
    data_args.task_name = data_args.task_name.lower()
    if data_args.task_name not in processors:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    if data_args.support_file is None or data_args.train_file is None or data_args.dev_file is None:
        raise ValueError("Support/Train/Dev files are required")
    if training_args.do_predict and data_args.test_file is None:
        raise ValueError("Test file is required for `--do_predict`")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformersX.utils.logging.set_verbosity(log_level)
    transformersX.utils.logging.enable_default_handler()
    transformersX.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Preprocessing the raw_datasets
    #
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    processor = processors[data_args.task_name]()
    dataset_dict = {
        "support": FewshotProcessor.to_datasets(processor.get_support_examples(data_args.support_file)),
        "train": FewshotProcessor.to_datasets(processor.get_train_examples(data_args.train_file, data_args.max_train_samples_per_label)),
        "dev": FewshotProcessor.to_datasets(processor.get_dev_examples(data_args.dev_file)),
    }
    if training_args.do_predict:
        dataset_dict["test"] = FewshotProcessor.to_datasets(processor.get_test_examples(data_args.test_file))
    raw_datasets = DatasetDict(dataset_dict)
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_set = set(raw_datasets["support"].unique("label"))
    label_list = sorted(list(label_set)) + sorted([l for l in raw_datasets["train"].unique("label") if l not in label_set]) # Let's sort labels for determinism
    label_to_id = {v: i for i, v in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForDualPassageEncoder.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    text1_key, text2_key = "text_a", "text_b"
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples[text1_key], padding=padding, max_length=max_seq_length, truncation=True)
        if text2_key in examples:
            result2 = tokenizer(examples[text2_key], padding=padding, max_length=max_seq_length, truncation=True)
            result = { k: np.stack([np.array(result[k]), np.array(result2[k])], axis=1).tolist() for k in result.keys() & result2.keys() }
        
        # Map labels to IDs
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Truncate datasets if max samples is specified
    support_dataset = raw_datasets["support"]
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    eval_dataset = raw_datasets["dev"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # You can define your custom compute_predicts function. It takes a `FewshotPrediction` object (a namedtuple with a
    # support_preds, support_labels, eval_preds and eval_labels field) and has to return an `EvalPrediction` object.
    def compute_predicts(p: FewshotPrediction):
        support_preds = p.support_preds[0] if isinstance(p.support_preds, tuple) else p.support_preds
        eval_preds = p.eval_preds[0] if isinstance(p.eval_preds, tuple) else p.eval_preds

        scores = np.einsum('ik,jk->ij', eval_preds, support_preds)
        indices = np.argmax(scores, axis=-1)[...,None]
        preds = p.support_labels[None,...].repeat(indices.shape[0], axis=0)
        preds = np.take_along_axis(preds, indices, axis=-1)
        preds = np.squeeze(preds, axis=-1)

        return EvalPrediction(predictions=preds, label_ids=p.eval_labels)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        label_ids = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
        return {"accuracy": (predictions == label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = FewshotTrainer(
        model=model,
        args=training_args,
        support_dataset=support_dataset,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_predicts=compute_predicts,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info(f"*** Evaluate {data_args.task_name} ***")

        metrics = trainer.evaluate(support_dataset=support_dataset, eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info(f"*** Predict {data_args.task_name} ***")

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("label") if "label" in predict_dataset else predict_dataset
        predictions = trainer.predict(support_dataset=support_dataset, test_dataset=predict_dataset, metric_key_prefix="predict").predictions

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {data_args.task_name} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-fewshot"}
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "fewshot"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"Fewshot {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()

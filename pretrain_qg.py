import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_metric
from nltk.tokenize import word_tokenize
from transformers import (
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed, AutoConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer)
from transformers.trainer_utils import get_last_checkpoint

from data_helper import read_data

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='t5-small',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default='cache/', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"},
    )
    task: Optional[str] = field(
        default='qg_agno',
        metadata={
            "help": "cloze2normal, normal2cloze, multi, qg, qg_agno"},
    )

    answer_aware: Optional[int] = field(
        default=0,
        metadata={"help": 'include answer during training?'},
    )

    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=48,
        metadata={"help": "Max input length for the target text"},
    )

    is_debug_mode: Optional[int] = field(
        default=-1,
        metadata={"help": "training on local machine?"},
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )


def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # set seed & init logger
    set_loggers(training_args)

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

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        model_max_length=512,
        # use_auth_token=True if model_args.use_auth_token else None,
    )

    train_dataset, valid_dataset = read_data(data_args, tokenizer)

    if data_args.is_debug_mode == 1:
        config = AutoConfig.from_pretrained('t5-small')
        config.d_ff = 64
        config.d_kv = 2
        config.d_model = 16
        # config.hidden_size = 16
        # config.num_attention_heads = 2
        config.num_layers = 2
        config.num_heads = 2
        config.num_decoder_layers = 2
        # config.num_hidden_layers = 2
        model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id
    )

    metric_squad = load_metric("squad")
    metric_bleu = load_metric("bleu")
    metric_rouge = load_metric("rouge")
    metric_meteor = load_metric('meteor')

    def postprocess_bleu(preds, labels):
        preds = [word_tokenize(pred) for pred in preds]
        labels = [[word_tokenize(label)] for label in labels]

        return preds, labels

    def postprocess_squad(preds, labels):
        preds = [{'prediction_text': p, 'id': i} for i, p in enumerate(preds)]
        labels = [{'answers': {'answer_start': [-1], 'text': [r]}, 'id': i} for i, r in enumerate(labels)]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # F1 & EM evaluations
        decoded_preds_tmp, decoded_labels_tmp = postprocess_squad(decoded_preds, decoded_labels)
        result_f1_em = metric_squad.compute(predictions=decoded_preds_tmp, references=decoded_labels_tmp)

        # Meteor evaluation
        result_meteor = metric_meteor.compute(predictions=decoded_preds, references=decoded_labels)

        # Extract a few results from ROUGE
        decoded_preds, decoded_labels = postprocess_bleu(decoded_preds, decoded_labels)
        result_rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = {key: value.mid.fmeasure * 100 for key, value in result_rouge.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result_rouge["gen_len"] = np.mean(prediction_lens)
        result_rouge = {k: round(v, 4) for k, v in result_rouge.items()}

        # Extract bleu
        result_bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result_bleu = {
            'bleu': result_bleu['bleu'] * 100,
            # 'bleu_precisions': result_bleu['precisions']
        }

        super_dict = {}  # uses set to avoid duplicates
        for d in [result_f1_em, result_rouge, result_bleu, result_meteor]:
            for k, v in d.items():
                super_dict[k] = v

        return super_dict

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(valid_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            valid_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(valid_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(valid_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                if data_args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    tmp_labels = np.where(predict_results.label_ids != -100, predict_results.label_ids,
                                          tokenizer.pad_token_id)
                else:
                    tmp_labels = predict_results.label_ids
                ground_truth = tokenizer.batch_decode(
                    tmp_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                pretty_results = [(grnd.strip(), pred.strip()) for grnd, pred in zip(ground_truth, predictions)]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for item in pretty_results:
                        writer.write('ground: {}\n'.format(item[0]))
                        writer.write('pred: {}\n'.format(item[1]))
                        writer.write('-' * 50 + '\n')

    return results


def set_loggers(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed
    set_seed(training_args.seed)
    # Set project name
    os.environ["WANDB_PROJECT"] = "qg-baselines"

    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)


if __name__ == "__main__":
    main()

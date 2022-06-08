import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser, T5Tokenizer

from data_collator import T2TDataCollator

# from dataset_helper import MyCustomSQuAD, load_squad, MyCustomDS
# from pretrain_qg import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    valid_file_path: str = ield(
        metadata={"help": "Path for cached valid dataset"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "num_beams to use for decoding"}
    )
    max_decoding_length: Optional[int] = field(
        default=48,
        metadata={"help": "maximum length for decoding"}
    )
    output_path: Optional[str] = field(
        default="runs/hypothesis.txt",
        metadata={"help": "path to save the generated questions."}
    )

    is_debug_mode: Optional[int] = field(
        default=-1,
        metadata={"help": "run on local machine ?"}
    )


def get_predictions(model, tokenizer, data_loader,
                    num_beams=6,
                    max_length=32,
                    length_penalty=1,
                    no_repeat_ngram_size=4):
    model.to(device)
    inputs = []
    references = []
    predictions = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader):
            outs = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

            inputs.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']])
            references.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']])
            predictions.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outs])

    return inputs, predictions, references


def main():
    parser = HfArgumentParser((EvalArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # valid_dataset = load_dataset(args.valid_file_path, tokenizer)
    valid_dataset = torch.load(args.valid_file_path)

    collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=args.model_type,
        mode="inference"
    )
    loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, collate_fn=collator)

    inputs, predictions, references = get_predictions(
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        num_beams=args.num_beams,
        max_length=args.max_decoding_length
    )

    with open(args.output_path, 'w') as outfile:
        for i, j, k in zip(predictions, references, inputs):
            outfile.write('context: {}\n ref_q: {}\n gen_q: {}\n'.format(k, j, i))
            outfile.write('-' * 50 + '\n')

    new_predictions = []
    new_references = []

    for p in predictions:
        new_predictions.append(p.strip().split(' '))

    for r in references:
        new_references.append([r.strip().split(' ')])

    bleu = corpus_bleu(new_references, new_predictions)
    print('BLEU: {0}'.format(bleu))


if __name__ == "__main__":
    main()

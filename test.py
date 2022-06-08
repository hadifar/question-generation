from datasets import load_metric
from nltk.tokenize import word_tokenize
import numpy as np

predictions = [
    "hello there general kenobi 4213213!"  # tokenized prediction of the first sample
]
references = [
    "hello there general kenobi 4213213"
]

squad_metric = load_metric("squad")


def postprocess_text2(preds, labels):
    preds = [{'prediction_text': p, 'id': i} for i, p in enumerate(preds)]
    labels = [{'answers': {'answer_start': [-1], 'text': [r]}, 'id': i} for i, r in enumerate(labels)]
    return preds, labels


predictions, references = postprocess_text2(predictions, references)
results = squad_metric.compute(predictions=predictions, references=references)
print(results)
#
# def postprocess_text(preds, labels):
#     preds = [word_tokenize(pred) for pred in preds]
#     labels = [[word_tokenize(label)] for label in labels]
#
#     return preds, labels
#
#
# predictions, references = postprocess_text(predictions, references)
# metric1 = load_metric("rouge")
# # metric2 = load_metric("rouge")
# # metric3 = load_metric("squad")
# # metric4 = load_metric("EM")
#
# for bleu in [metric1]:
#     result = bleu.compute(predictions=predictions, references=references)
#     # Extract a few results from ROUGE
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
#
#     prediction_lens = [np.count_nonzero(pred != 0) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     print(result)
#     print('-'*100)

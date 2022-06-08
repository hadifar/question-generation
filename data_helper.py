from torch.utils.data import Dataset
import json

from itertools import chain


def tokenized_data(source_text, target_text, data_args, tokenizer):
    padding = "max_length" if data_args.pad_to_max_length else False
    model_inputs = tokenizer(source_text, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_text, max_length=data_args.max_target_length, padding=padding, truncation=True)

    if data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def process_normal2cloze(data, data_args, tokenizer):
    all_questions = [item['question'] for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt & eos token
        inp_txt = 'normal2cloze: ' + q['normal_format']
        out_txt = q['cloze_format']
        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return MyCustomDS(model_inputs.data)


def process_cloze2normal(data, data_args, tokenizer):
    all_questions = [item['question'] for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt & eos token
        inp_txt = 'cloze2normal: ' + q['cloze_format']
        out_txt = q['normal_format']
        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return MyCustomDS(model_inputs.data)


def process_multi(data, data_args, tokenizer):
    all_questions = [item['question'] for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt & eos token
        inp_txt = 'cloze2normal: ' + q['cloze_format']
        out_txt = q['normal_format']
        source_text.append(inp_txt)
        target_text.append(out_txt)

        inp_txt = 'normal2cloze: ' + q['normal_format']
        out_txt = q['cloze_format']

        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return MyCustomDS(model_inputs.data)


def process_data(data, data_args, tknizer):
    if data_args.task == 'cloze2normal':
        return process_cloze2normal(data, data_args, tknizer)
    elif data_args.task == 'normal2cloze':
        return process_normal2cloze(data, data_args, tknizer)
    elif data_args.task == 'multi':
        return process_multi(data, data_args, tknizer)
    else:
        raise Exception('fu')


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def read_data(data_args, tokenizer):
    train_data = read_json_file(data_args.train_file_path)
    valid_data = read_json_file(data_args.valid_file_path)

    if data_args.is_debug_mode == 1:
        train_data = train_data[:3]
        valid_data = valid_data[:3]

    train_ds = process_data(train_data, data_args, tokenizer)
    valid_ds = process_data(valid_data, data_args, tokenizer)

    return train_ds, valid_ds


class MyCustomDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# class MyCustomSQuAD(Dataset):
#     def __init__(self, data, tknzr, max_source_len=512, max_target_len=48):
#         self.max_source_length = max_source_len
#         self.max_target_length = max_target_len
#         self.tokenizer = tknzr
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         question = self.data[idx]['contents'] + ' </s>'
#         context = self.data[idx]['context'] + ' </s>'
#
#         source_encoding = self.tokenizer.encode_plus(
#             context,
#             max_length=self.max_source_length,
#             padding='max_length',
#             pad_to_max_length=True,
#             truncation=True,
#             # return_length='pt'
#         )
#
#         target_encoding = self.tokenizer.encode_plus(
#             question,
#             max_length=self.max_target_length,
#             padding='max_length',
#             pad_to_max_length=True,
#             truncation=True,
#             # return_tensors='pt'
#         )
#
#         tmp = {'source_ids': source_encoding['input_ids'],
#                'target_ids': target_encoding['input_ids'],
#                'attention_mask': source_encoding['attention_mask']
#                }
#
#         return tmp
#
#
# class SimpleOpenStax(Dataset):
#     def __init__(self, dataset_path, tknzr, max_source_len=512, max_target_len=48):
#         self.max_source_length = max_source_len
#         self.max_target_length = max_target_len
#         self.tokenizer = tknzr
#         with open(dataset_path) as outfile:
#             self.data = json.load(outfile)
#
#         # self.data.append(t['retrieve'][:self.topk])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # if (random.random() < 0.5):
#         question = self.data[idx]['question_text'] + ' </s>'
#
#         context = []
#         for i, sent in enumerate(self.data[idx]['answer']['ans_txt']):
#             if i in self.data[idx]['answer']['sentence_answer']:
#                 sent = "<hl> " + " ".join(sent) + " <hl>"
#             else:
#                 sent = " ".join(sent)
#             context.append(sent)
#
#         context = " ".join(context) + ' </s>'
#
#         # question = self.data[idx]['contents']
#         # sentences = self.data[idx]['context']
#         # prob_list = softmax(np.array(self.data[idx]['scores']) / temperature)
#         #
#         # draw = choice(range(len(prob_list)), size=1, p=prob_list)[0]
#         #
#         # sentences[draw] = '<hl> ' + sentences[draw] + ' <hl>'
#         # context = " ".join(sentences)
#
#         source_encoding = self.tokenizer.encode_plus(
#             context,
#             max_length=self.max_source_length,
#             padding='max_length',
#             pad_to_max_length=True,
#             truncation=True,
#             # return_length='pt'
#         )
#
#         target_encoding = self.tokenizer.encode_plus(
#             question,
#             max_length=self.max_target_length,
#             padding='max_length',
#             pad_to_max_length=True,
#             truncation=True,
#             # return_tensors='pt'
#         )
#
#         tmp = {'source_ids': source_encoding['input_ids'],
#                'target_ids': target_encoding['input_ids'],
#                'attention_mask': source_encoding['attention_mask'],
#                # 'conf_score': prob_list[draw]
#                }
#
#         return tmp

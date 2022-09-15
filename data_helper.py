from torch.utils.data import Dataset
import json
import nltk
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
    all_questions = [item for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt & eos token
        # if data_args.answer_aware == 1:
        #     ans_txt = q['question']['question_choices'][q['answer']['ans_choice']]
        #     inp_txt = 'answer: {} normal2cloze: {}'.format(ans_txt, q['question']['normal_format'])
        # else:
        inp_txt = 'normal2cloze: ' + q['question']['normal_format']

        out_txt = q['question']['cloze_format']
        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return CustomDS(model_inputs.data)


def process_cloze2normal(data, data_args, tokenizer):
    all_questions = [item for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt
        # if data_args.answer_aware == 1:
        #     ans_txt = q['question']['question_choices'][q['answer']['ans_choice']]
        #     inp_txt = 'answer: {} cloze2normal: {}'.format(ans_txt, q['question']['cloze_format'])
        # else:
        inp_txt = 'cloze2normal: ' + q['question']['cloze_format']

        out_txt = q['question']['normal_format']

        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return CustomDS(model_inputs.data)


def process_multi(data, data_args, tokenizer, split):
    all_questions = [item for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:

        if split == 'train':

            inp_txt = 'cloze2normal: ' + q['question']['cloze_format']
            out_txt = q['question']['normal_format']
            source_text.append(inp_txt)
            target_text.append(out_txt)

            inp_txt = 'normal2cloze: ' + q['question']['normal_format']
            out_txt = q['question']['cloze_format']
            source_text.append(inp_txt)
            target_text.append(out_txt)

        else:
            if data_args.task == 'multi_cloze2normal':
                # if data_args.answer_aware == 1:
                #     ans_txt = q['question']['question_choices'][q['answer']['ans_choice']]
                #     inp_txt = 'answer: {} cloze2normal: {}'.format(ans_txt, q['question']['cloze_format'])
                # else:
                inp_txt = 'cloze2normal: ' + q['question']['cloze_format']
                out_txt = q['question']['normal_format']
                source_text.append(inp_txt)
                target_text.append(out_txt)
            elif data_args.task == 'multi_normal2cloze':
                # if data_args.answer_aware == 1:
                #     ans_txt = q['question']['question_choices'][q['answer']['ans_choice']]
                #     inp_txt = 'answer: {} normal2cloze: {}'.format(ans_txt, q['question']['normal_format'])
                # else:
                inp_txt = 'normal2cloze: ' + q['question']['normal_format']
                out_txt = q['question']['cloze_format']

                source_text.append(inp_txt)
                target_text.append(out_txt)
            else:
                raise Exception("task is not found {multiway conversion}")

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return CustomDS(model_inputs.data)


def process_qg_openstax(data, data_args, tokenizer):
    all_questions = [item for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt & eos token
        ans_txt = q['question']['question_choices'][q['answer']['ans_choice']]
        context = q['hl_sentences']
        inp_txt = 'answer: {} context: {}'.format(ans_txt, context)
        out_txt = q['question']['normal_format']
        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return CustomDS(model_inputs.data)


def process_qg_agno_openstax(data, data_args, tokenizer):
    all_questions = [item for item in list(chain(*[d['questions'] for d in data]))]
    source_text = []
    target_text = []
    for q in all_questions:
        # add prompt & eos token
        # ans_txt = q['question']['question_choices'][q['answer']['ans_choice']]
        # context = q['hl_sentences']
        context = q['hl_context'].replace('<hl>', '')
        inp_txt = 'context: {}'.format(context)
        out_txt = q['question']['normal_format']
        source_text.append(inp_txt)
        target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return CustomDS(model_inputs.data)


def process_qg_squad(data, data_args, tokenizer):
    source_text = []
    target_text = []
    for wiki_page in data:
        for p in wiki_page['paragraphs']:
            question_list = p['qas']
            context = p['context']
            # total_n_docs += 1
            # avg_d_len += len(nltk.tokenize.word_tokenize(context))
            sentences = nltk.tokenize.sent_tokenize(context)
            for qobject in question_list:

                if qobject['is_impossible']:
                    continue
                question = qobject['question']
                answer = qobject['answers'][0]['text']

                q_beg_indx = qobject['answers'][0]['answer_start']
                b = 0
                e = 0

                tmp_con = []
                for s in sentences:
                    e += len(s)
                    if b <= q_beg_indx <= e:
                        # answer sentence

                        tmp_con.append("<hl> " + s + " <hl>")
                    else:
                        tmp_con.append(s)
                    # else:
                    #     pass
                    # print('unmatch')
                    # non answer sentence
                    b += len(s)

                inp_txt = 'answer: {} context: {}'.format(answer, " ".join(tmp_con))
                out_txt = question

                source_text.append(inp_txt)
                target_text.append(out_txt)

    model_inputs = tokenized_data(source_text, target_text, data_args, tokenizer)
    return CustomDS(model_inputs.data)


def process_data(data, data_args, tknizer, split):
    if data_args.task == 'cloze2normal':
        return process_cloze2normal(data, data_args, tknizer)
    elif data_args.task == 'normal2cloze':
        return process_normal2cloze(data, data_args, tknizer)
    elif data_args.task == 'multi_cloze2normal':
        return process_multi(data, data_args, tknizer, split)
    elif data_args.task == 'multi_normal2cloze':
        return process_multi(data, data_args, tknizer, split)
    elif data_args.task == 'qg':
        return process_qg_openstax(data, data_args, tknizer)
    elif data_args.task == 'qg_agno':
        return process_qg_agno_openstax(data, data_args, tknizer)
    else:
        raise Exception('fu')


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def read_data(data_args, tokenizer):
    if data_args.valid_file_path.endswith('dev-v2.0.json'):
        train_data = read_json_file(data_args.train_file_path)
        valid_data = read_json_file(data_args.valid_file_path)['data']
        if data_args.is_debug_mode == 1:
            train_data = train_data[:3]
            valid_data = valid_data[:3]
        train_ds = process_data(train_data, data_args, tokenizer, 'train')
        valid_ds = process_qg_squad(valid_data, data_args, tokenizer)
    else:
        train_data = read_json_file(data_args.train_file_path)
        valid_data = read_json_file(data_args.valid_file_path)

        if data_args.is_debug_mode == 1:
            train_data = train_data[:3]
            valid_data = valid_data[:3]

        train_ds = process_data(train_data, data_args, tokenizer, 'train')
        valid_ds = process_data(valid_data, data_args, tokenizer, 'valid')

    return train_ds, valid_ds


class CustomDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

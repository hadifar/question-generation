# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
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

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import nltk

nltk.download('punkt')

import nlp

_CITATION = """\
@InProceedings{hadifar:dataset,
title = {A Common Crawl for QA},
authors={Amir Hadifar
},
year={2021}
}
"""

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care. 
"""

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]


class QGConfig(nlp.BuilderConfig):

    def __init__(self, qg_format="highlight", **kwargs):
        super(QGConfig, self).__init__(**kwargs)
        self.qg_format = qg_format


class QG(nlp.GeneratorBasedBuilder):
    _URL = "data/cc_qg"
    # _URL1 = "https://cloud.ilabt.imec.be/index.php/s/DFQALHziH4RkXpY/download"
    # _URL2 = "https://cloud.ilabt.imec.be/index.php/s/AZEidxEe37t5iAW/download"

    _URL1 = "https://cloud.ilabt.imec.be/index.php/s/gWxFaAbWQyeobrX/download"
    _URL2 = "https://cloud.ilabt.imec.be/index.php/s/zgMbtxtN6nxBPFf/download"
    _DEV_FILE = "valid_qa_pairs.txt"
    _TRAINING_FILE = "train_qa_pairs.txt"

    BUILDER_CONFIGS = [
        QGConfig(
            name=f"{format_}_qg_format",
            description="Plain text",
            version="1.0.0",
            qg_format=format_
        )
        for format_ in QG_FORMATS
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "source_text": nlp.Value("string"),
                    "target_text": nlp.Value("string"),
                    "task": nlp.Value("string"),
                    # "similarity_score": nlp.Value("float32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self._URL1),
            "dev": os.path.join(self._URL2),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _get_correct_alignement(self, context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """

        gold_text = answer
        start_idx = context.find(answer)
        # print(context)
        # print(answer)
        # print(start_idx)
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        # elif context[start_idx - 1:end_idx - 1] == gold_text:
        #     return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        # elif context[start_idx - 2:end_idx - 2] == gold_text:
        #     return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()

    def process_e2e_qg(self, instance):
        source_text = f"generate questions: {instance['context'].strip()}"
        # questions = [instance['question']]
        # target_text = " {sep_token} ".join(questions)
        target_text = f"{instance['question']} {{sep_token}}"
        return {"source_text": source_text,
                "target_text": target_text,
                "task": "e2e_qg",
                # "similarity_score": random.uniform(0, 1)
                }

    def process_e2e_qg_v2(self, instance):
        source_text = f"generate questions: {instance['context'].strip()}"
        answer_text = instance['rule_question'][0]['A'].strip()

        target_text = f"{instance['question']} {{sep_token}} {answer_text} {{sep_token}}"

        return {"source_text": source_text,
                "target_text": target_text,
                "task": "e2e_qg_v2",
                # "similarity_score": random.uniform(0, 1)
                }

    def process_qg(self, instance):
        answer_text = instance['rule_question'][0]['A'].strip()
        context = instance['context'].strip()
        # sim_score = instance['rule_question'][0]['score']

        if self.config.qg_format == "prepend":
            que_gen_input = f"answer: {answer_text}  context: {context}"
        elif self.config.qg_format == "highlight":
            start_pos, end_pos = self._get_correct_alignement(context, answer_text)
            que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        else:
            raise ValueError()

        que_gen_target = f"{instance['question']}"
        return {"source_text": que_gen_input,
                "target_text": que_gen_target,
                "task": "qg",
                # 'similarity_score': sim_score
                }

    def process_ans_ext(self, instance):
        # answer_text =
        # context = instance['context'].strip()
        answer_text = instance['rule_question'][0]['A'].strip()
        context = instance['context'].strip()
        sentences = nltk.sent_tokenize(context)

        input_text = "{hl_token} " + " {hl_token} ".join(sentences) + " {hl_token}"
        output_text = answer_text + " {sep_token}"

        return {"source_text": f"extract answers:" + input_text,
                "target_text": output_text,
                "task": "ans_ext",
                # 'similarity_score': sim_score
                }

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        count = 0
        tasks = ['qa', 'qg', 'ans_ext', 'e2e_qg', 'e2e_qg_v2']
        with open(filepath) as f:
            for line in f:
                instance = json.loads(line)

                if 'ans_ext' in tasks:
                    yield count, self.process_ans_ext(instance)
                    count += 1

                if 'qg' in tasks:
                    yield count, self.process_qg(instance)
                    count += 1

                if 'e2e_qg' in tasks:
                    yield count, self.process_e2e_qg(instance)
                    count += 1

                if 'e2e_qg_v2' in tasks:
                    yield count, self.process_e2e_qg_v2(instance)
                    count += 1

                # else:
                #     raise NotImplementedError('The the process example is not implemented ...')

# -*- coding: utf-8 -*-
#
# Copyright 2021 Amir Hadifar. All Rights Reserved.
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
# ==============================================================================

from pipelines import pipeline

nlp = pipeline('e2e-qg-v2',model='t5-small-e2e-qg-v2-hl-plus-rules', tokenizer='t5_qg_tokenizer')

text = "Easy! Jen Lancaster came to town! And a co-worker (and fellow WNBA member) is a huge fan and knew about the event and asked me if I wanted to go. And as I should have kept reading the two books I was in the middle of, I tried to ignore this book that promised to be funny and a fast read, which I was totally in the mood for. I could hear it calling to me. I got one of my other books and put it on top of the Lancaster, hoping that would silence it, and encourage me to pick up and finish the half-read one. But funny won the day. I read half Saturday, finished Sunday!"

print(nlp(text))

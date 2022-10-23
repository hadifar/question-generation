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

nlp = pipeline('e2e-qg-openstax', model='hadifar/openstax_qg_agno', tokenizer='hadifar/openstax_qg_agno')

while True:
    text = input("Enter your text:")
    reply_dic = nlp(text)
    print('Question:', reply_dic)

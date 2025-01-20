# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from typing import List


class OpenAIEmbeddingsWrapper:
    def __init__(self, openai_embeddings):
        self.openai_embeddings = openai_embeddings

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        # Use OpenAI's embed_query for queries
        return list(map(np.array, self.openai_embeddings.embed_documents(queries)))

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        # Use OpenAI's embed_documents for documents
        return list(map(np.array, self.openai_embeddings.embed_documents(documents)))
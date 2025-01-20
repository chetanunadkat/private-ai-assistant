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

import logging
import os
from typing import Any, Dict, List
from traceback import print_exc

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field

from src.retrievers.base import BaseExample
from src.common.utils import (
    get_text_splitter,
    get_vectorstore,
    get_embedding_model,
    get_llm,
    create_vectorstore_langchain,
    del_docs_vectorstore_langchain,
    get_config,
    get_docs_vectorstore_langchain,
    get_prompts,
)

logger = logging.getLogger(__name__)
document_embedder = get_embedding_model()
text_splitter = None
settings = get_config()
prompts = get_prompts()
vector_db_top_k = int(os.environ.get(f"VECTOR_DB_TOPK", 40))

try:
    vectorstore = create_vectorstore_langchain(document_embedder=document_embedder)
except Exception as e:
    vectorstore = None
    logger.info(f"Unable to connect to vector store during initialization: {e}")


class UnstructuredRetriever(BaseExample):
    def ingest_docs(self, filepath: str, filename: str) -> None:
        """Ingests documents to the VectorDB.
        It's called when the POST endpoint of `/documents` API is invoked.

        Args:
            filepath (str): The path to the document file.
            filename (str): The name of the document file.

        Raises:
            ValueError: If there's an error during document ingestion or the file format is not supported.
        """
        if not filename.endswith((".txt", ".pdf", ".md")):
            raise ValueError(f"{filename} is not a valid Text, PDF or Markdown file")
        try:
            # Load raw documents from the directory
            _path = filepath
            raw_documents = UnstructuredFileLoader(_path).load()

            if raw_documents:
                global text_splitter
                # Get text splitter instance, it is selected based on environment variable APP_TEXTSPLITTER_MODELNAME
                # tokenizer dimension of text splitter should be same as embedding model
                if not text_splitter:
                    text_splitter = get_text_splitter()

                # split documents based on configuration provided
                documents = text_splitter.split_documents(raw_documents)
                vs = get_vectorstore(vectorstore, document_embedder)
                # ingest documents into vectorstore
                vs.add_documents(documents)
            else:
                logger.warning("No documents available to process!")
        except Exception as e:
            logger.error(f"Failed to ingest document due to exception {e}")
            raise ValueError("Failed to upload document. Please upload an unstructured text document.")


    def document_search(self, content: str, num_docs: int, user_id: str = None, conv_history: Dict[str, str] = {}) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on the query.
        Args:
            content: Query string
            num_docs: Number of documents to return
            user_id: Optional user ID for tracking
            conv_history: Optional conversation history
        Returns:
            List of relevant documents
        """
        try:
            logger.info(f"Searching relevant document for the query: {content}")
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs is None:
                logger.error("Vector store not initialized properly")
                return []
            docs = vs.similarity_search_with_score(content, k=num_docs)
            return [{"content": doc[0].page_content, "source": os.path.basename(doc[0].metadata.get("source", "")), "score": doc[1]} for doc in docs]
        except Exception as e:
            logger.warning(f"Failed to generate response due to exception {str(e)}")
            return []


    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store.
        It's called when the GET endpoint of `/documents` API is invoked.

        Returns:
            List[str]: List of filenames ingested in vectorstore.
        """
        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs:
                return get_docs_vectorstore_langchain(vs)
        except Exception as e:
            logger.error(f"Vectorstore not initialized. Error details: {e}")
        return []


    def delete_documents(self, filenames: List[str]) -> bool:
        """Delete documents from the vector index.
        It's called when the DELETE endpoint of `/documents` API is invoked.

        Args:
            filenames (List[str]): List of filenames to be deleted from vectorstore.
        """
        try:
            # Get vectorstore instance
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs:
                return del_docs_vectorstore_langchain(vs, filenames)
        except Exception as e:
            logger.error(f"Vectorstore not initialized. Error details: {e}")
        return False

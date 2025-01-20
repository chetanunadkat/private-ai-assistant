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

""" Retriever pipeline for extracting data from structured information"""
import logging
import os
from typing import Any, Dict, List
from urllib.parse import urlparse
import pandas as pd
from src.retrievers.structured_data.vaanaai.vaana_base import VannaWrapper
from src.retrievers.base import BaseExample
from src.common.utils import get_config

logger = logging.getLogger(__name__)
settings = get_config()

# Load the vaana_client
vaana_client = VannaWrapper()
# Connect to the Postgress DB
app_database_url = get_config().database.url

# Parse the URL
parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')

# Extract host and port
host = parsed_url.hostname
port = parsed_url.port

vaana_client.connect_to_postgres(
    host=parsed_url.hostname, 
    dbname=os.getenv("POSTGRES_DB",'customer_data'), 
    user=os.getenv('POSTGRES_USER', 'postgres_readonly'), 
    password= os.getenv('POSTGRES_PASSWORD', 'readonly_password'), 
    port=parsed_url.port
    )
# Do Training from static schmea
vaana_client.do_training(method="schema")

class CSVChatbot(BaseExample):
    """RAG example showcasing CSV parsing using Vaana AI Agent"""

    def ingest_docs(self, filepath: str, filename: str):
        """Ingest documents to the VectorDB."""

        raise NotImplementedError("Canonical RAG only supports document retrieval")

    def document_search(self, content: str, num_docs: int, user_id: str = None, conv_history: Dict[str, str] = {}) -> List[Dict[str, Any]]:
        """Execute a Document Search."""

        logger.info("Using document_search to fetch response from database as text")

        # Do training if needed
        vaana_client.do_training(method="ddl")

        try:
            if not user_id:
                logger.warning("No User ID provided")
                return [{"content": "Please provide a valid User ID to search for data."}]

            logger.info(f"Querying data for user_id: {user_id} with content: {content}")
            result_df = vaana_client.ask_query(question=content, user_id=user_id)
            
            logger.info("Result Data Frame: %s", result_df)
            
            # Handle various error cases
            if result_df is None:
                return [{"content": "No data found for the given query."}]
            
            if isinstance(result_df, str) and result_df == "not valid sql":
                return [{"content": "Unable to generate a valid SQL query for your request. Please rephrase your question."}]
            
            if isinstance(result_df, pd.DataFrame):
                if result_df.empty:
                    return [{"content": "No matching records found for your query."}]
                if result_df.shape == (1, 1) and not bool(result_df.iloc[0, 0]):
                    return [{"content": "The query returned no results."}]
                
                # Format DataFrame as a string with proper formatting
                result_str = result_df.to_string(index=False)
                return [{"content": result_str}]
            
            return [{"content": str(result_df).strip()}]
            
        except Exception as e:
            logger.error("An error occurred during document search: %s", str(e))
            return [{"content": f"An error occurred while processing your request: {str(e)}"}]

    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store."""
        logger.error("get_documents not implemented")
        return True

    def delete_documents(self, filenames: List[str]):
        """Delete documents from the vector index."""
        logger.error("delete_documents not implemented")
        return True

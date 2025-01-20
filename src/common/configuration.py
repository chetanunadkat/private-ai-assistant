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

"""The definition of the application configuration."""
from src.common.configuration_wizard import ConfigWizard, configclass, configfield


@configclass
class VectorStoreConfig(ConfigWizard):
    """Configuration class for the Vector Store connection.

    :cvar name: Name of vector store
    :cvar url: URL of Vector Store
    """

    name: str = configfield(
        "name",
        default="milvus", # supports pgvector, milvus
        help_txt="The name of vector store",
    )
    url: str = configfield(
        "url",
        default="http://milvus:19530", # for pgvector `pgvector:5432`
        help_txt="The host of the machine running Vector Store DB",
    )
    nlist: int = configfield(
        "nlist",
        default=64, # IVF Flat milvus
        help_txt="Number of cluster units",
    )
    nprobe: int = configfield(
        "nprobe",
        default=16, # IVF Flat milvus
        help_txt="Number of units to query",
    )

@configclass
class DatabaseConfig(ConfigWizard):
    """Configuration class for the Database connection.

    :cvar name: Name of Database
    :cvar url: URL of Database
    :cvar config: config shared to database
    """

    from dataclasses import field
    name: str = configfield(
        "name",
        default="postgres", # supports redis, postgres
        help_txt="The name of database",
    )
    url: str = configfield(
        "url",
        default="postgres:5432", # for redis `redis:6379`
        help_txt="The host of the machine running database",
    )
    config: str = configfield(
        "config",
        default=field(default_factory=""),
        help_txt="Any configuration needs to be shared can be shared as dict",
    )

@configclass
class CheckpointerConfig(ConfigWizard):
    """Configuration class for the Database connection.

    :cvar name: Name of checkpointer database
    :cvar url: URL of checkpointer database
    :cvar config: config shared to database
    """

    from dataclasses import field
    name: str = configfield(
        "name",
        default="postgres", # supports inmemory, postgres
        help_txt="The name of database",
    )
    url: str = configfield(
        "url",
        default="postgres:5432", # for redis `redis:6379`
        help_txt="The host of the machine running database",
    )
    config: str = configfield(
        "config",
        default=field(default_factory=""),
        help_txt="Any configuration needs to be shared can be shared as dict",
    )

@configclass
class CacheConfig(ConfigWizard):
    """Configuration class for the Vector Store connection.

    :cvar name: Name of Cache
    :cvar url: URL of Cache
    :cvar config: config shared to Cache
    """

    from dataclasses import field
    name: str = configfield(
        "name",
        default="redis", # supports redis
        help_txt="The name of vector store",
    )
    url: str = configfield(
        "url",
        default="redis:6379", # for redis `redis:6379`
        help_txt="The host of the machine running cache",
    )
    config: str = configfield(
        "config",
        default=field(default_factory=""),
        help_txt="Any configuration needs to be shared can be shared as dict",
    )

@configclass
class LLMConfig(ConfigWizard):
    """Configuration class for the llm connection.

    :cvar model_name: The name of the OpenAI model.
    :cvar model_engine: The type of model engine (openai).
    """

    model_name: str = configfield(
        "model_name",
        default="gpt-4o-2024-05-13",
        help_txt="The name of the OpenAI model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="openai",
        help_txt="The server type of the hosted model. Allowed values are openai",
    )

@configclass
class TextSplitterConfig(ConfigWizard):
    """Configuration class for the Text Splitter.

    :cvar chunk_size: Chunk size for text splitter in characters.
    :cvar chunk_overlap: Number of characters to overlap between chunks.
    """

    chunk_size: int = configfield(
        "chunk_size",
        default=1000,
        help_txt="Chunk size for text splitting in characters.",
    )
    chunk_overlap: int = configfield(
        "chunk_overlap",
        default=200,
        help_txt="Number of characters to overlap between chunks.",
    )


@configclass
class EmbeddingConfig(ConfigWizard):
    """Configuration class for the Embeddings.

    :cvar model_name: The name of the embedding model.
    :cvar model_engine: The type of model engine (openai or huggingface).
    :cvar dimensions: The dimensions of the embeddings.
    """

    model_name: str = configfield(
        "model_name",
        default="text-embedding-3-small",
        help_txt="The name of the embedding model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="openai",
        help_txt="The server type of the hosted model. Allowed values are openai, huggingface",
    )
    dimensions: int = configfield(
        "dimensions",
        default=1536,
        help_txt="The required dimensions of the embedding model. Currently utilized for vector DB indexing.",
    )

@configclass
class RetrieverConfig(ConfigWizard):
    """Configuration class for the Retrieval pipeline.

    :cvar top_k: Number of relevant results to retrieve.
    :cvar score_threshold: The minimum confidence score for the retrieved values to be considered.
    """

    top_k: int = configfield(
        "top_k",
        default=4,
        help_txt="Number of relevant results to retrieve",
    )
    score_threshold: float = configfield(
        "score_threshold",
        default=0.25,
        help_txt="The minimum confidence score for the retrieved values to be considered",
    )

@configclass
class AppConfig(ConfigWizard):
    """Configuration class for the application.

    :cvar vector_store: The configuration of the vector db connection.
    :type vector_store: VectorStoreConfig
    :cvar llm: The configuration of the backend llm server.
    :type llm: LLMConfig
    :cvar text_splitter: The configuration for text splitter
    :type text_splitter: TextSplitterConfig
    :cvar embeddings: The configuration for embeddings
    :type embeddings: EmbeddingConfig
    """

    vector_store: VectorStoreConfig = configfield(
        "vector_store",
        env=False,
        help_txt="The configuration of the vector db connection.",
        default=VectorStoreConfig(),
    )
    database: DatabaseConfig = configfield(
        "database",
        env=False,
        help_txt="The configuration of the database connection.",
        default=DatabaseConfig(),
    )
    checkpointer: CheckpointerConfig = configfield(
        "checkpointer",
        env=False,
        help_txt="The configuration of the checkpointer.",
        default=CheckpointerConfig(),
    )
    cache: CacheConfig = configfield(
        "cache",
        env=False,
        help_txt="The configuration of the cache connection.",
        default=CacheConfig(),
    )
    llm: LLMConfig = configfield(
        "llm",
        env=False,
        help_txt="The configuration for the server hosting the Large Language Models.",
        default=LLMConfig(),
    )
    text_splitter: TextSplitterConfig = configfield(
        "text_splitter",
        env=False,
        help_txt="The configuration for text splitter.",
        default=TextSplitterConfig(),
    )
    embeddings: EmbeddingConfig = configfield(
        "embeddings",
        env=False,
        help_txt="The configuration of embedding model.",
        default=EmbeddingConfig(),
    )
    retriever: RetrieverConfig = configfield(
        "retriever",
        env=False,
        help_txt="The configuration of the retriever pipeline.",
        default=RetrieverConfig(),
    )
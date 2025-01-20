Okay, here are the specific changes required for the **unstructured data retriever** part of the project when switching to OpenAI and removing NVIDIA-specific components, while still using Milvus as vectorstore:

**Changes Focus:**

*   Replacing NVIDIA's embedding model with OpenAI's embedding model.
*   Keeping Milvus as the vector store.
*   Removing the reranker.
*   Updating the necessary configurations and code to reflect these changes.

**File-by-File Changes for Unstructured Data Handling:**

**1. `ai-virtual-assistant/deploy/helm/charts/retriever-canonical/values.yaml`**

*   **Changes:** We need to update the environment variables for the `retriever-canonical` container to use OpenAI's API key and model names, and to remove NVIDIA-specific configurations.
*   **Steps:**
    1. Find the `env:` section under `applicationSpecs.canonical-deployment.containers.retriever-canonical`.
    2. Change the following environment variables:

        ```yaml
        - name: APP_LLM_MODELNAME
          value: "gpt-4o-2024-05-13" # Update to gpt-4o
        - name: APP_LLM_MODELENGINE
          value: "openai" # Update to OpenAI
        - name: APP_LLM_SERVERURL
          value: "" # Update to remove the NVIDIA server URL
        - name: APP_EMBEDDINGS_MODELNAME
          value: "text-embedding-3-small"
        - name: APP_EMBEDDINGS_MODELENGINE
          value: "openai"
        - name: APP_EMBEDDINGS_SERVERURL
          value: ""
        - name: APP_TEXTSPLITTER_MODELNAME
          value: "gpt-4o-2024-05-13"
        - name: OPENAI_API_KEY
          value: "" # Add your OpenAI API Key here
        ```

    3. Remove the following environment variables as they are related to the NVIDIA models and reranker:

        ```yaml
        - name: APP_RANKING_MODELNAME
        - name: APP_RANKING_MODELENGINE
        - name: APP_RANKING_SERVERURL
        - name: NVIDIA_API_KEY #Also remove this
        ```
    4. Update the `COLLECTION_NAME` if you want to use a different name for your collection in Milvus.

**2. `ai-virtual-assistant/src/retrievers/unstructured_data/chains.py`**

*   **Changes:**
    *   Update imports to use `OpenAIEmbeddings`.
    *   Modify `create_vectorstore_langchain` if needed to be compatible with `OpenAIEmbeddings`.
    *   Modify `document_search` to remove reranker logic and update the search process to be compatible with the new embedding model.
    *   Update `ingest_docs` to ensure compatibility with `OpenAIEmbeddings` and the modified vector store setup.
    *   Update `get_text_splitter` to use `RecursiveCharacterTextSplitter`
    *   Update `import` statements

*   **Steps:**

    1. **Update Imports:**

        ```python
        # Add these imports
        from langchain_openai import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Milvus
        ```

    2. **Modify `create_vectorstore_langchain`:**

        ```python
        def create_vectorstore_langchain(document_embedder, collection_name: str = "") -> VectorStore:
            \"\"\"Create the vector db index for langchain.\"\"\"

            config = get_config()

            if config.vector_store.name == "milvus":
                # Updated for Milvus with OpenAI embeddings
                connection_args = {"uri": config.vector_store.url, "token": "your_milvus_api_key_or_credentials"}  # Replace with Milvus credentials
                vectorstore = Milvus(
                    embedding_function=document_embedder,
                    collection_name=collection_name,
                    connection_args=connection_args
                )
            else:
                raise ValueError(f"{config.vector_store.name} vector database is not supported")
            logger.info("Vector store created and saved.")
            return vectorstore
        ```

        **Note:**
        *   You'll need to replace `"your_milvus_api_key_or_credentials"` with your actual Milvus connection details.
        *   Ensure that the `collection_name` is correctly set for your Milvus setup.

    3. **Modify `document_search`:**

        ```python
        def document_search(self, content: str, num_docs: int, conv_history: Dict[str, str] = {}) -> List[Dict[str, Any]]:
            \"\"\"Search for the most relevant documents for the given search parameters.
            It's called when the `/search` API is invoked.

            Args:
                content (str): Query to be searched from vectorstore.
                num_docs (int): Number of similar docs to be retrieved from vectorstore.
            \"\"\"

            logger.info(f"Searching relevant document for the query: {content}")

            try:
                vs = get_vectorstore(vectorstore, document_embedder)
                if vs is None:
                    logger.error(f"Vector store not initialized properly. Please check if the vector db is up and running")
                    raise ValueError()

                # Use similarity_search for Milvus
                docs = vs.similarity_search_with_score(content, k=num_docs)

                resp = []
                for doc, score in docs:
                    resp.append(
                        {
                            "source": os.path.basename(doc.metadata.get("source", "")),
                            "content": doc.page_content,
                            "score": score,  # Assuming Milvus returns a score
                        }
                    )
                return resp

            except Exception as e:
                logger.warning(f"Failed to generate response due to exception {e}")
                print_exc()

            return []
        ```
      4. **Modify `get_text_splitter`:**
          ```python
          def get_text_splitter() -> RecursiveCharacterTextSplitter:
              \"\"\"Return the text splitter instance from langchain.\"\"\"
              return RecursiveCharacterTextSplitter(
                  chunk_size=get_config().text_splitter.chunk_size,
                  chunk_overlap=get_config().text_splitter.chunk_overlap,
              )
          ```
      5. **Modify `ingest_docs`:**
          ```python
          def ingest_docs(self, filepath: str, filename: str) -> None:
              \"\"\"Ingests documents to the VectorDB.
              It's called when the POST endpoint of `/documents` API is invoked.

              Args:
                  filepath (str): The path to the document file.
                  filename (str): The name of the document file.

              Raises:
                  ValueError: If there's an error during document ingestion or the file format is not supported.
              \"\"\"
              if not filename.endswith((".txt", ".pdf", ".md")):
                  raise ValueError(f"{filename} is not a valid Text, PDF or Markdown file")
              try:
                  # Load raw documents from the directory
                  _path = filepath
                  raw_documents = UnstructuredFileLoader(_path).load()

                  if raw_documents:
                      # Get text splitter instance
                      text_splitter = get_text_splitter()

                      # split documents based on configuration provided
                      documents = text_splitter.split_documents(raw_documents)

                      # Ensure document_embedder is initialized before this call
                      global document_embedder
                      if document_embedder is None:
                          document_embedder = get_embedding_model()

                      vs = get_vectorstore(vectorstore, document_embedder)
                      # Ingest documents into Milvus
                      vs.add_documents(documents)
                  else:
                      logger.warning("No documents available to process!")
              except Exception as e:
                  logger.error(f"Failed to ingest document due to exception {e}")
                  raise ValueError("Failed to upload document. Please check the document format.")
          ```

**13. `ai-virtual-assistant/deploy/compose/docker-compose.yaml`**

*   **Changes:** Remove or comment out NVIDIA NIM services and update environment variables.
*   **Steps:**
    1. **Comment out or remove** the following services:
        *   `nemollm-inference`
        *   `nemollm-embedding`
        *   `ranking-ms`
    2. **Update Environment Variables:**
        *   In the `agent-chain-server` service, update the environment variables:
            *   Remove `APP_LLM_SERVERURL`, `APP_EMBEDDINGS_MODELNAME`, `APP_EMBEDDINGS_MODELENGINE`, `APP_EMBEDDINGS_SERVERURL`, `APP_RANKING_MODELNAME`, `APP_RANKING_MODELENGINE`, and `APP_RANKING_SERVERURL`.
            *   Add or update:
                ```
                APP_LLM_MODELNAME

*   **Changes:** Remove or comment out NVIDIA NIM services and update environment variables.
*   **Steps:**
    2. **Update Environment Variables (Continued):**
        *   In the `unstructured-retriever` service, update the environment variables:
            *   Remove `APP_LLM_SERVERURL`, `APP_EMBEDDINGS_SERVERURL`, `APP_RANKING_MODELNAME`, `APP_RANKING_MODELENGINE`, and `APP_RANKING_SERVERURL`.
            *   Add or update:
                ```
                APP_LLM_MODELNAME: gpt-4o-2024-05

```yaml
                APP_LLM_MODELENGINE: openai
                APP_EMBEDDINGS_MODELNAME: text-embedding-3-small
                APP_EMBEDDINGS_MODELENGINE: openai
                OPENAI_API_KEY: ${OPENAI_API_KEY}
                ```
        *   Ensure that `APP_VECTORSTORE_URL` points to your Milvus instance, and `APP_VECTORSTORE_NAME` is set to `milvus`.
        *   Ensure that `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` are correctly set for your Postgres database if you're using the structured data retriever.

**Here's an example of how the `unstructured-retriever` service definition might look in your `docker-compose.yaml` after these changes:**

```yaml
  unstructured-retriever:
    container_name: unstructured-retriever
    image: nvcr.io/nvidia/blueprint/aiva-customer-service-unstructured-retriever:1.1.0
    build:
      # Set context to repo's root directory
      context: ../../
      dockerfile: src/retrievers/Dockerfile
      args:
        # Build args, used to copy relevant directory inside the container
        EXAMPLE_PATH: 'src/retrievers/unstructured_data'
    # start the server on port 8081
    command: --port 8081 --host 0.0.0.0 --workers 1
    environment:
      EXAMPLE_PATH: 'src/retrievers/unstructured_data'
      APP_VECTORSTORE_URL: "http://milvus:19530" # Assuming milvus service is named 'milvus'
      APP_VECTORSTORE_NAME: "milvus"
      APP_LLM_MODELNAME: "gpt-4o-2024-05-13"
      APP_LLM_MODELENGINE: "openai"
      APP_EMBEDDINGS_MODELNAME: "text-embedding-3-small"
      APP_EMBEDDINGS_MODELENGINE: "openai"
      APP_TEXTSPLITTER_MODELNAME: "gpt-4o-2024-05-13" # Using OpenAI model for text splitting
      APP_TEXTSPLITTER_CHUNKSIZE: 506
      APP_TEXTSPLITTER_CHUNKOVERLAP: 200
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      COLLECTION_NAME: "unstructured_data" # Or any name you prefer for your Milvus collection
      APP_RETRIEVER_TOPK: 4
      APP_RETRIEVER_SCORETHRESHOLD: 0.25
      VECTOR_DB_TOPK: 20
      LOGLEVEL: INFO
    ports:
    - "8086:8081"
    expose:
    - "8081"
    shm_size: 5gb
    depends_on:
    - milvus
```
**14. Update the helm chart values:**
Update the values in values.yaml file under each microservice, that we changed above, to reflect the changes done for docker compose.

**Building and Deploying**

1. **Build Docker Images:**

    *   Since you have modified the application code, you'll need to rebuild the Docker images for the affected services. You can use the following command from the `ai-virtual-assistant` directory:

        ```bash
        docker compose -f deploy/compose/docker-compose.yaml build
        ```
2. **Deploy with Docker Compose:**

    *   Start the application stack using:

        ```bash
        docker compose -f deploy/compose/docker-compose.yaml up -d
        ```

**Testing and Refinement:**

*   **Thoroughly test** the application using the UI or the API endpoints.
*   **Use various queries,** including those that involve structured and unstructured data, to ensure all components are working correctly.
*   **Monitor logs** for errors or unexpected behavior.
*   **Adjust prompts:** You might need to iteratively refine the prompts in `prompt.yaml` to get the best performance from the OpenAI models.

**Additional Notes:**

*   **Error Handling:** Make sure to add appropriate error handling in your code to gracefully handle potential issues with the OpenAI API (e.g., rate limits, API errors).
*   **Cost Management:** Be mindful of the cost of using OpenAI's APIs. Monitor your usage and consider implementing caching mechanisms to reduce costs.
*   **Security:** Securely store your OpenAI API key. Avoid hardcoding it directly in your code or configuration files. Use environment variables or a secrets management system.

**Important Considerations for Milvus:**

*   **Index Type:** When creating the Milvus collection in `create_vectorstore_langchain`, make sure you choose an index type that is optimized for CPU usage if you're not using a GPU. `IVF_FLAT` or `HNSW` are generally good choices for CPU.
*   **OpenAI Embeddings Dimensionality:** OpenAI's `text-embedding-3-small` model produces embeddings with 1536 dimensions by default. You may need to adjust the configuration of your Milvus collection (e.g. change in number of dimensions) to match this. If you use `text-embedding-3-large` then update it to 3072.

This detailed guide should provide a solid starting point for your junior developer. The most important aspects are replacing the NVIDIA components with their OpenAI equivalents, adapting the prompts, and ensuring that Milvus is configured to work efficiently with the new embeddings. Remember to encourage questions and provide support throughout the process. I am here to help further if needed.

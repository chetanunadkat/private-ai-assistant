**File-by-File Change Guide (Revised):**

**1. `ai-virtual-assistant/deploy/helm/charts/agent-services/values.yaml`** (⏳ Postponed)
*   **Changes:**  Update environment variables for the `agent-services-container` to use OpenAI's API key and model name. Remove NVIDIA specific configurations.
*   **Steps:**
    1. Find the `env:` section under `applicationSpecs.agent-services-deployment.containers.agent-services-container`.
    2. Change the following environment variables:

        ```yaml
        - name: APP_LLM_MODELNAME
          value: "gpt-4o-2024-05-13"
    3. Remove the following environment variables as they are related to NVIDIA's NIMs:
        ```yaml
          - name: APP_LLM_SERVERURL
          - name: APP_EMBEDDINGS_MODELNAME
          - name: APP_EMBEDDINGS_MODELENGINE
          - name: APP_EMBEDDINGS_SERVERURL
          - name: APP_RANKING_MODELNAME
          - name: APP_RANKING_MODELENGINE
          - name: APP_RANKING_SERVERURL
        ```
    4. Add the following environment variable for your OpenAI API key:
        ```yaml
        - name: OPENAI_API_KEY
          value: "" # Add your OpenAI API Key here
        ```
    5. Modify the environment variable `APP_LLM_MODELENGINE`
        ```
         - name: APP_LLM_MODELENGINE
           value: "openai"
        ```

**2. `ai-virtual-assistant/src/agent/requirements.txt`** ✅
*   **Changes:** Replace `langchain-nvidia-ai-endpoints` with `langchain-openai`, remove unnecessary dependencies, and add any new ones required for OpenAI.
*   **Steps:**
    1. ✅ Remove the line: `langchain-nvidia-ai-endpoints==0.3.5`
    2. ✅ Add the line: `langchain-openai`
    3. ✅ Remove these lines:
        ```
        langgraph-checkpoint-postgres==2.0.0
        psycopg-binary==3.2.3
        ```
    4. ✅ Ensure that `langchain` is present in the `requirements.txt` file. If not add it. (version: `langchain==0.3.0`)
    5. ✅ Add `openai` to the list of the dependencies

**3. `ai-virtual-assistant/src/agent/utils.py`** ✅
*   **Changes:** Modify `get_llm`, `get_embedding_model`, `create_vectorstore_langchain`, `get_vectorstore`, `get_checkpointer`, `remove_state_from_checkpointer` functions to use OpenAI's APIs and remove NVIDIA-specific code.
*   **Steps:**
    1. ✅ **Imports:**
        ```python
        # Remove
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
        from langchain_core.runnables import RunnableLambda
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        import nest_asyncio
        import asyncio

        # Add
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain.memory import ConversationBufferMemory
        ```

    2. ✅ **Modify `get_llm`:**
        ```python
        @utils_cache
        @lru_cache
        def get_llm(**kwargs) -> ChatOpenAI:
            \"\"\"Create the LLM connection.\"\"\"
            settings = get_config()
            logger.info(f"Using OpenAI model: {settings.llm.model_name}")
            return ChatOpenAI(model=settings.llm.model_name, openai_api_key=os.environ["OPENAI_API_KEY"], **kwargs)
        ```

    3. ✅ **Modify `get_embedding_model`:**
        ```python
        @lru_cache
        def get_embedding_model() -> OpenAIEmbeddings:
           \"\"\"Create the embedding model.\"\"\"
           settings = get_config()
           logger.info(f"Using OpenAI embedding model: {settings.embeddings.model_name}")
           return OpenAIEmbeddings(model=settings.embeddings.model_name, api_key=os.environ["OPENAI_API_KEY"])
        ```
    4. ✅ **Modify `create_vectorstore_langchain`:**

        ```python
        def create_vectorstore_langchain(document_embedder, collection_name: str = "") -> FAISS:
            \"\"\"Create the vector db index for langchain.\"\"\"

            config = get_config()

            if config.vector_store.name == "faiss":
                vectorstore = FAISS.from_documents([], document_embedder) # Pass empty documents list
            else:
                raise ValueError(f"{config.vector_store.name} vector database is not supported")
            logger.info("Vector store created and saved.")
            return vectorstore
        ```
        Since we are now using in-memory FAISS, we need to update this method, we are passing an empty list of document as we don't need them anymore.

    5. ✅ **Modify `get_vectorstore`:**

        ```python
        def get_vectorstore(vectorstore, document_embedder) -> FAISS:
            \"\"\"
            Send a vectorstore object.
            If a Vectorstore object already exists, the function returns that object.
            Otherwise, it creates a new Vectorstore object and returns it.
            \"\"\"
            if vectorstore is None:
                return create_vectorstore_langchain(document_embedder)
            return vectorstore
        ```
    6. ✅ **Modify `get_checkpointer`:**
        Change the checkpointer to be memory based rather than Postgres based.

        ```python
        async def get_checkpointer() -> tuple:
             settings = get_config()

             if settings.checkpointer.name == "inmemory":
                 print(f"Using MemorySaver as checkpointer")
                 return MemorySaver(), None
             else:
                 raise ValueError(f"Only inmemory and postgres is supported chckpointer type")
        ```

    7. ✅ **Modify `remove_state_from_checkpointer`:**
        Update the function to clear the checkpointer for in-memory.

        ```python
        def remove_state_from_checkpointer(session_id):
            settings = get_config()
            if settings.checkpointer.name == "inmemory":
                # For inmemory checkpointer we just need to reinitialize the memory
                global memory
                memory = MemorySaver()
            else:
                # For other supported checkpointer(i.e. inmemory) we don't need cleanup
                pass

        ```

    8. ✅ **Modify the imports in `main.py`:**

        ```python
        # Remove these imports
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        import nest_asyncio

        # Add these imports
        from langchain_openai import ChatOpenAI
        from langchain.memory import ConversationBufferMemory
        ```
    9. ✅ **Modify the `get_checkpointer` call:**
        Change the call for `get_checkpointer` function
        ```python
        async def get_checkpointer():
             global memory
             memory = MemorySaver()
        ```
    10. ✅ **Remove unnecessary code in `main.py`**:
         Remove the following lines of code, as we are not using `AsyncPostgresSaver` and `nest_asyncio` anymore.

         ```python
         # Allow multiple async loop togeather
         # This is needed to create checkpoint as it needs async event loop
         # TODO: Move graph build into a async function and call that to remove nest_asyncio
         import nest_asyncio
         nest_asyncio.apply()

         # To run the async main function
         import asyncio

         pool = None

         # TODO: Remove pool as it's not getting used
         # WAR: It's added so postgres does not close it's session
         async def get_checkpoint():
             global memory, pool
             memory, pool = await get_checkpointer()

         asyncio.run(get_checkpoint())
         ```
         Also update the call for `get_checkpointer` to `get_checkpoint`.

         ```python
         # Compile
         graph = builder.compile(checkpointer=memory,
                                 interrupt_before=["return_processing_sensitive_tools"],
                                 #interrupt_after=["ask_human"]
                                 )
         ```

    11. ✅ **Remove:** `get_product_name` function as we are using the `vaana.ai` for interacting with the structured data.

**4. `ai-virtual-assistant/src/common/utils.py`** ✅
*   **Changes:** Update imports and functions to use OpenAI instead of NVIDIA
*   **Steps:**
    1. ✅ **Update imports:**
        ```python
        # Remove
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

        # Add
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        ```
    2. ✅ **Update `get_llm`:**
        ```python
        @utils_cache
        @lru_cache()
        def get_llm(**kwargs) -> LLM | SimpleChatModel:
            settings = get_config()
            if settings.llm.model_engine == "openai":
                return ChatOpenAI(
                    model=settings.llm.model_name,
                    temperature=kwargs.get('temperature', None),
                    top_p=kwargs.get('top_p', None),
                    max_tokens=kwargs.get('max_tokens', None),
                    api_key=os.environ["OPENAI_API_KEY"]
                )
        ```
    3. ✅ **Update `get_embedding_model`:**
        ```python
        @lru_cache
        def get_embedding_model() -> Embeddings:
            settings = get_config()
            if settings.embeddings.model_engine == "openai":
                return OpenAIEmbeddings(
                    model=settings.embeddings.model_name,
                    api_key=os.environ["OPENAI_API_KEY"]
                )
        ```
    4. ✅ Remove `get_ranking_model` function as it's no longer needed

**5. `ai-virtual-assistant/src/common/configuration.py`** ✅
*   **Changes:** Update configuration classes for OpenAI
*   **Steps:**
    1. ✅ **Update LLMConfig:**
        ```python
        @configclass
        class LLMConfig(ConfigWizard):
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
        ```
    2. ✅ **Update EmbeddingConfig:**
        ```python
        @configclass
        class EmbeddingConfig(ConfigWizard):
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
                help_txt="The required dimensions of the embedding model.",
            )
        ```
    3. ✅ Remove RankingConfig class
    4. ✅ Update RetrieverConfig to remove NVIDIA-specific fields

**6. `ai-virtual-assistant/src/retrievers/structured_data/chains.py`** ✅
*   **Changes:** Replace the usage of `NvidiaLLM` with `VannaWrapper`, leveraging its ability to work with OpenAI models. Update the `get_purchase_history` function to use `VannaWrapper` for data retrieval.
*   **Steps:**
    1. ✅ **Update Imports:**

        ```python
        #from src.common.utils import get_llm, get_prompts, get_config
        from vaanaai.vaana_base import VannaBase
        from vaanaai.vaana_llm import VannaLLM

        ```
    2. ✅ **Modify `get_purchase_history`:**
        ```python
        def get_purchase_history(user_id: str) -> str:
            \"\"\"Retrieves the recent return and order details for a user,
            including order ID, product name, status, relevant dates, quantity, and amount.\"\"\"

            # Initialize VannaWrapper
            vanna_model = VannaWrapper()

            # Use VannaWrapper to generate SQL query
            sql = vanna_model.generate_sql(question=f"Get purchase history for user ID {user_id}")

            # Execute the SQL query
            if vanna_model.is_sql_valid(sql, user_id):
               result = vanna_model.run_sql(sql)
            else:
               return "Invalid SQL query generated for the given user ID."

            # Returning result as a list of dictionaries
            if isinstance(result, pd.DataFrame):
               return result.to_dict(orient='records')
            else:
               return []

        ```
    3. ✅ **Update `structured_rag`:** Modify this function to use the updated `get_purchase_history` and format the output appropriately. You will likely need to adapt how the output of `get_purchase_history` is used in subsequent steps.

**7. `ai-virtual-assistant/src/retrievers/structured_data/vaanaai/vaana_base.py`** ✅
*   **Changes:** Replace the initialization of `NvidiaLLM` with `VannaWrapper` and utilize its methods for interacting with OpenAI models.
*   **Steps:**
    1. ✅ **Update Initialization:**
        ```python
        class VannaWrapper(Milvus_VectorStore, VannaLLM):
            def __init__(self, config=None):
                logger.info("Initializing MyVanna with VannaLLM and Milvus_VectorStore")
                document_embedder = get_embedding_model()
                emb_function = OpenAIEmbeddingsWrapper(document_embedder)
                settings = get_config()
                if settings.vector_store.name == "milvus":
                    milvus_db_url = settings.vector_store.url
                milvus_client = MilvusClient(uri=milvus_db_url)
                Milvus_VectorStore.__init__(self, config={"embedding_function": emb_function, "milvus_client": milvus_client})
                VannaLLM.__init__(self, config=config)
        ```
    2. ✅ **Modify `do_training`:** Update the `do_training` method to utilize the new `VannaWrapper` class for training. Ensure that the training data is properly formatted for use with OpenAI models.

**8. `ai-virtual-assistant/src/retrievers/structured_data/vaanaai/vaana_llm.py`** ✅
*   **Changes:** Replace the `NvidiaLLM` class with a `VannaWrapper` class that utilizes `ChatOpenAI`.
*   **Steps:**
    1. ✅ **Update Imports:**
        ```python
        from langchain_openai import ChatOpenAI
        ```
    2. ✅ **Modify `VannaLLM`:** Update the class to use `ChatOpenAI` for generating SQL queries and submitting prompts.

        ```python
        class VannaLLM(VannaBase):
            def __init__(self, config=None):
                default_llm_kwargs = {"temperature": 0.2, "top_p": 0.7, "max_tokens": 1024}
                self.model = ChatOpenAI(model="gpt-4o-2024-05-13", api_key=os.environ["OPENAI_API_KEY"], **default_llm_kwargs)

            def system_message(self, message: str) -> any:
                return {"role": "system", "content": message}

            def user_message(self, message: str) -> any:
                return {"role": "user", "content": message}

            def assistant_message(self, message: str) -> any:
                return {"role": "assistant", "content": message}

            def generate_sql(self, question: str, **kwargs) -> str:
                # Use the super generate_sql
                sql = super().generate_sql(question, **kwargs)
                # Replace "\_" with "_"
                sql = sql.replace("\\_", "_")

                return sql

            def submit_prompt(self, prompt, **kwargs) -> str:
                response = self.model.invoke(prompt)
                return response.content
        ```
        We are using ChatOpenAI here with model="gpt-4o-2024-05-13"

**9. `ai-virtual-assistant/src/retrievers/structured_data/vaanaai/utils.py`** ✅
*   **Changes:**  Update the `NVIDIAEmbeddingsWrapper` class to use OpenAI embeddings.
*   **Steps:**
    1. ✅ **Modify `NVIDIAEmbeddingsWrapper`:**

        ```python
        from langchain_openai import OpenAIEmbeddings

        class OpenAIEmbeddingsWrapper:
             def __init__(self, openai_embeddings):
                 self.openai_embeddings = openai_embeddings

             def encode_queries(self, queries: List[str]) -> List[np.array]:
                 # Use OpenAI's embed_query for queries
                 return list(map(np.array, self.openai_embeddings.embed_documents(queries)))

             def encode_documents(self, documents: List[str]) -> List[np.array]:
                 # Use OpenAI's embed_documents for documents
                 return list(map(np.array, self.openai_embeddings.embed_documents(documents)))
        ```

**10. `ai-virtual-assistant/deploy/compose/docker-compose.yaml`** ✅
*   **Changes:** Remove or comment out NVIDIA NIM services and update environment variables.
*   **Steps:**
    1. ✅ **Comment out or remove** the following services:
        *   `nemollm-inference`
        *   `nemollm-embedding`
        *   `ranking-ms`
    2. ✅ **Update Environment Variables:**
        *   In the `agent-chain-server` and `unstructured-retriever` services, remove the environment variables related to NVIDIA models and servers:
            *   `APP_LLM_MODELNAME` (within agent-chain-server only)
            *   `APP_LLM_MODELENGINE`
            *   `APP_LLM_SERVERURL`
            *   `APP_EMBEDDINGS_MODELNAME`
            *   `APP_EMBEDDINGS_MODELENGINE`
            *   `APP_EMBEDDINGS_SERVERURL`
            *   `APP_RANKING_MODELNAME`
            *   `APP_RANKING_MODELENGINE`
            *   `APP_RANKING_SERVERURL`
        *   Add the following environment variable to both services:
            ```
            OPENAI_API_KEY: ${OPENAI_API_KEY}
            ```
        *   In the `agent-chain-server` service, add the following environment variables if they don't exist:
            ```
            APP_LLM_MODELNAME: gpt-4o-2024-05-13
            APP_LLM_MODELENGINE: openai
            ```
        *   In the `unstructured-retriever` service, add the following environment variables if they don't exist:
            ```
            APP_EMBEDDINGS_MODELNAME: text-embedding-3-small
            APP_EMBEDDINGS_MODELENGINE: openai
            ```

**11. Building and Deploying** ⏳

*   **Build Docker Images:** After making the code changes, rebuild your Docker images for the `agent-chain-server`, `unstructured-retriever`, and `structured-retriever` services.
*   **Deploy with Docker Compose:**
    ```bash
    docker compose -f deploy/compose/docker-compose.yaml up -d
    ```

**12. Testing:** ⏳

*   Thoroughly test the application using the UI or the API endpoints.
*   Use various queries, including those that involve structured and unstructured data, to ensure all components are working correctly.
*   Monitor logs for errors or unexpected behavior.
*   Evaluate the quality of the responses from the OpenAI models and adjust prompts or parameters as needed.

### Docker Compose Changes ✅
✅ Updated docker-compose.yaml:
- Removed NVIDIA services:
  - nemollm-inference
  - nemollm-embedding
  - ranking-ms
- Updated service configurations:
  - agent-chain-server
  - unstructured-retriever
  - structured-retriever
  - analytics-server
- Modified environment variables for OpenAI
- Updated dependencies and health checks
 




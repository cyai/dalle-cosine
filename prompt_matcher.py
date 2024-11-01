from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from uuid import uuid4


class PromptMatcher:
    def __init__(
        self,
        collection_name="treasure_collection",
        model_name="sentence-transformers/all-mpnet-base-v2",
        storage_path="/tmp/langchain_qdrant",
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.client = QdrantClient(path=storage_path)

        self.collection_name = collection_name
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def store_prompts(self, prompts):
        documents = [
            Document(page_content=prompt, metadata={"prompt_id": prompt_id})
            for prompt_id, prompt in prompts.items()
        ]
        uuids = [str(uuid4()) for _ in range(len(documents))]

        self.vector_store.add_documents(documents=documents, ids=uuids)

    def match_prompt(self, user_prompt, user_prompt_id):
        # results = self.vector_store.similarity_search(user_prompt, k=1)
        retriever = self.vector_store.as_retriever(
            earch_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        results = retriever.invoke(input=user_prompt, k=1)

        print(results)
        if results:
            best_match = results[0]

            matched_prompt_id = best_match.metadata.get("prompt_id")

            if matched_prompt_id == user_prompt_id:
                return "Success"
            else:
                return "Fail"
        return "Fail"

import os

import numpy as np
import ollama
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# load data
def load_data(data_fd):
    all_texts = []
    for filename in sorted(os.listdir(data_fd)):
        if filename.endswith(".pdf"):
            file_path = os.path.abspath(os.path.join(data_fd, filename))
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    all_texts.append((i + 1, file_path, text))
    return all_texts


# chunk data
def chunk_data(
    data, chunk_size=1024, chunk_overlap=200
):  # 1024 tokens, 200 tokens overlap
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    tokenizer = splitter._tokenizer
    chunks = []
    for page_label, file_path, text in data:
        for chunk in splitter._split_text(
            text,
            chunk_size
            - len(tokenizer(f"page_label: {page_label}\nfile_path: {file_path}")),
        ):
            chunks.append((page_label, file_path, chunk))

    return chunks


model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


# embed chunks
def augment_chunk(page_label, file_path, text):
    return f"page_label: {page_label}\nfile_path: {file_path}\n\n{text}"


def prompt_text(text):
    return f"search_document:{text}"


def embed_texts(texts):
    prompted_texts = [prompt_text(text) for text in texts]
    embeddings = model.encode(prompted_texts, normalize_embeddings=True)
    return embeddings, prompted_texts


# embed query
def prompt_query(query):
    return f"search_query:{query}"


def embed_query(query):
    prompted_query = prompt_query(query)
    embeddings = model.encode([prompted_query], normalize_embeddings=True)
    return embeddings, prompted_query


def retrieval(emb_q, embeds, augmented_chunks, top_k=2):
    similarities = np.dot(embeds, emb_q.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = [augmented_chunks[i] for i in top_indices]
    top_similarities = similarities[top_indices]
    retrieval_result = list(zip(top_chunks, top_similarities))
    return retrieval_result


def make_chat_messages(query, retrieval_result):
    SYSTEM_MESSAGE = (
        "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided"
        " context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given"
        " context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information"
        " ...' or anything along those lines."
    )

    qa_message_template = (
        "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context"
        " information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "
    )

    context_str = "\n\n".join([r[0] for r in retrieval_result])
    qa_message = qa_message_template.format(context_str=context_str, query_str=query)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": qa_message},
    ]

    return messages


def llm_response(messages, model="llama3.1:latest", temperature=0.75, num_ctx=3900):
    return ollama.chat(
        model="llama3.1:latest",
        messages=messages,
        stream=False,
        tools=None,
        options={"temperature": temperature, "num_ctx": num_ctx},
    )


if __name__ == "__main__":
    data_fd = "./data"
    docs = load_data(data_fd)
    chunks = chunk_data(docs)
    augmented_chunks = [augment_chunk(*chunk) for chunk in chunks]
    embeddings, texts = embed_texts(augmented_chunks)
    query = "what is nougat and how does it work?"
    embeddingQ, textQ = embed_query(query)
    retrieval_result = retrieval(embeddingQ, embeddings, augmented_chunks, top_k=4)
    messages = make_chat_messages(query, retrieval_result)
    response = llm_response(messages, temperature=0.01)
    print(
        "Query:\n{query}\n\nRetrieval:\n{ret}\n\nAnswer:\n{answer}".format(
            query=query,
            ret="\n\n".join([r[0] for r in retrieval_result]),
            answer=response["message"]["content"],
        )
    )

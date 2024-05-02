import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import huggingface_hub
from langchain import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from langchain import PromptTemplate, LLMChain
import re
import PyPDF2
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import math


def load_model(model_id):
    print(f"Loading Model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir="./model/",
        use_auth_token="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",
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     cache_dir="./model/",
    #     torch_dtype=torch.float16,
    #     trust_remote_code=True,
    #     load_in_8bit=True,
    #     use_auth_token="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",
    # )

    # generation_config = GenerationConfig.from_pretrained(model_id)

    # # Create a pipeline for text generation
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_length=4096,
    #     top_p=0.95,
    #     repetition_penalty=1.15,
    #     generation_config=generation_config,
    # )

    # local_llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0})
    # print("Local LLM Loaded")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return tokenizer, tokenizer, embedding_model


def clean_document_nemerics(text):  # Cleans the text
    return re.sub(r"(?:\b|(?<=\s))\w{1}\b|[\d\W]+", " ", text).lower()


def files_from_Stramlit(files):
    full_String = ""
    for file in files:
        pdf_reader = PyPDF2.PdfReader(file)
        # Extract the content
        for page in range(len(pdf_reader.pages)):
            full_String += pdf_reader.pages[page].extract_text()

    cleaned_string = clean_document_nemerics(full_String)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,  # similar token len in overlap of text between chunks
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(cleaned_string)

    return chunks, cleaned_string


# def load_documents_and_chunks(directory):
#     documents = []
#     for item_path in glob(directory + "*.pdf"):
#         loader = PyPDFLoader(item_path)
#         documents.extend(loader.load())

#     documents = [i.page_content for i in documents]
#     full_string = " ".join(documents)
#     cleaned_string = clean_document_nemerics(full_string)

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2048,
#         chunk_overlap=128,  # similar token len in overlap of text between chunks
#         is_separator_regex=False,
#     )

#     chunks = text_splitter.split_text(cleaned_string)

#     return chunks, cleaned_string


def calc_tokens(text, tokenizer):
    return len(tokenizer.tokenize(text))


def cluster_sentences(
    sentences, embedding_model, distance_threshold=1.3
):  # AgglomerativeClustering for vector clustering
    sentence_embeddings = embedding_model.encode(sentences)

    clustering_model = AgglomerativeClustering(
        distance_threshold=distance_threshold, n_clusters=None, linkage="ward"
    )
    clustering_model.fit(sentence_embeddings)

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(clustering_model.labels_):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(sentences[sentence_id])

    return [cluster for cluster in clustered_sentences.values()]


def redistribution(listoflistofchunks, tokenizer):
    limit = 1024
    redistributed = []
    for index, chunks in enumerate(listoflistofchunks):
        tokens = calc_tokens("\n".join(chunks), tokenizer)

        print(index, tokens)

        if tokens < limit:
            redistributed.extend([chunks])
        else:
            print(
                f"chunk {index} of tokens {tokens} is splitted into {math.ceil(tokens/limit)} parts"
            )
            partitionlist = np.array_split(chunks, math.ceil(tokens / limit))
            # res = [list(x) for x in partitionlist]
            redistributed.extend([list(x) for x in partitionlist])
    return redistributed


def get_summary(files, LLM, tokenizer, embedding_model):
    chunks, cleaned_string = files_from_Stramlit(files)
    clusters = cluster_sentences(chunks, embedding_model)
    limit_clusters = redistribution(clusters, tokenizer)
    tosendlangchain = ["\n".join(i) for i in limit_clusters]

    res = []
    for i in tosendlangchain:
        res.append(calc_tokens(i, tokenizer))
    return res


# tokenizer, LLM, embedding_model = load_model("krthk/llama-2-7b-chat-finetuned") #Huggingface model id

# chunks, raw_text = load_documents_and_chunks("./docs/")

# clusters = cluster_sentences(chunks, embedding_model)

# limit_clusters = redistribution(clusters)

# tosendlangchain = ["\n".join(i) for i in limit_clusters]

# for i in tosendlangchain:
#   print(calc_tokens(i))

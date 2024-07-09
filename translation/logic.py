from datasets import load_dataset
from requests import post
from operator import itemgetter
from dotenv import load_dotenv
import os

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

CHROMA_DB_DIRECTORY = 'chroma_db/translate'

astar_fewshot_template = """
You are a post-editing translator editing inaccurate translations so that they are accurate.
Use the examples below to guide your translations. Do not include [Improved Malay] in the final translation.


Translate:
[English]: {eng_input}
[Malay]: {zsm_input}
[Improved Malay]:
"""

astar_fewshot_chain = None


def database_exists():
    print(f'Database exists: {os.path.exists(CHROMA_DB_DIRECTORY)}')
    return os.path.exists(CHROMA_DB_DIRECTORY)


def langchain_exists():
    print(f'Langchain exists: {astar_fewshot_chain is not None}')
    return astar_fewshot_chain is not None


def setup_backend():
    if not database_exists():
        build_database()
    if not langchain_exists():
        build_langchain()


def build_database():
    dataset = load_dataset(
        "facebook/flores", "eng_Latn-zsm_Latn", trust_remote_code=True)
    en_examples = dataset['dev']['sentence_eng_Latn'] + \
        dataset['devtest']['sentence_eng_Latn']
    my_examples = dataset['dev']['sentence_zsm_Latn'] + \
        dataset['devtest']['sentence_zsm_Latn']

    docs = [Document(s, metadata={'my_example': my_examples[i]})
            for i, s in enumerate(en_examples)]

    sbert_embed = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    db = Chroma.from_documents(
        documents=docs,
        embedding=sbert_embed,
        collection_name='translate',
        persist_directory=CHROMA_DB_DIRECTORY
    )


def build_langchain():
    """
    Runs text through ASTAR Translation and then through a few-shot prompt to improve the translation.

    RAG is used to retrieve similar examples from the database.
    The examples are translated to Malay using ASTAR Translation.
    The translated examples, together with the human-translated Malay sentences 
    are then used in a few-shot prompt to improve the translation.
    """

    def format_docs(docs):
        str = ""
        for doc in docs:
            doc: Document
            my_example = doc.metadata['my_example']
            en_example = doc.page_content

            # Run english example sentence through ASTAR Translation
            my_example_imperfect = _astar_translate(en_example)
            str += f"[English]: {en_example}\n[Malay]: {my_example_imperfect}\n[Improved Malay]: {my_example}\n"
        return str

    sbert_embed = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    db = Chroma(
        collection_name='translate',
        embedding_function=sbert_embed,
        persist_directory=CHROMA_DB_DIRECTORY
    )

    astar_fewshot_prompt = PromptTemplate.from_template(astar_fewshot_template)
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

    print(retriever.invoke('hi how are you'))

    llm = ChatOpenAI(model="gpt-4o")

    global astar_fewshot_chain
    astar_fewshot_chain = (
        {
            "examples": itemgetter("eng_input") | retriever | format_docs,
            "eng_input": itemgetter("eng_input"),
            "zsm_input": itemgetter("zsm_input")
        }
        | astar_fewshot_prompt
        | llm
        | StrOutputParser()
    )


# TODO: Improve translation time, do profiling?
def translate(text: str):
    translated = _astar_translate(text)

    post_edit_results = astar_fewshot_chain.invoke({
        'eng_input': text,
        'zsm_input': translated
    })
    return post_edit_results


def _astar_translate(text: str):
    api_endpoint = "http://54.255.172.249:5006/translator"
    headers = {"Content-Type": "application/json"}
    json = {
        "source": 'en_SG',
        "target": 'ms_SG',
        "query": text
    }

    response = post(api_endpoint, headers=headers, json=json)

    return ''.join([x['translatedText'] for x in response.json()['data']['translations']])

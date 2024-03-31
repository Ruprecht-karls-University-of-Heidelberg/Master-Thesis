from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,  # It will first select 200 characters from the text and then split it by the separator
                    # This is useful when the text is too long and we want to split it into smaller chunks for processing
    chunk_overlap=100  # The number of characters to overlap between chunks so it goes back the last 100 characters and then split
                    # This is useful when the separator is a sentence and we want to make sure that the sentence is 
                    # not split between two chunks. If there are more than one sentence in this overlap, it will split. 
                    # If there are more than one sentence in this overlap, it will split. Otherwise, it will continue 
                    # to take chunks until it finds a separator.
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

for doc in docs:
    print(doc.page_content)
    print("\n")
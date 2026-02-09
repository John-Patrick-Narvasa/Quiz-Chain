import chromadb
# from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# setting the environment

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# NOTE: Make sure the collection name is SAME as in fill_db.py
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

# NOTE: Can be CUSTOMIZED
user_query = input("What do you want to know about growing vegetables?\n\n")

# NOTE: Can be CUSTOMIZED based on the use case
results = collection.query(
    query_texts=[user_query],
    n_results=4
)

# print(results['documents'])
#print(results['metadatas'])

# client = OpenAI()
client = genai.Client()

# NOTE: Can be CUSTOMIZED
system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""+str(results['metadatas'])+"""
"""
# NOTE: Can ADD more data above as source if needed

#print(system_prompt)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents = f"{system_prompt}\n\nUser Question: {user_query}\n\nAnswer:",
)

print("\n\n---------------------\n\n")

# print(response.choices[0].message.content)
print(response.text)
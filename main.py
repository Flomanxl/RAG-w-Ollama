import ollama
import chromadb


# Function to intialise the extract from the file function.
def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to add documents to the collection
def add_document_to_collection(collection, document, doc_id):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=document)
    embedding = response["embedding"]
    collection.add(
        ids=[str(doc_id)],
        embeddings=[embedding],
        documents=[document]
    )

# Initialize chromadb client and create collection
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Function to process uploaded text file
def process_uploaded_file(file_path, collection):
    document = extract_text_from_file(file_path)
    add_document_to_collection(collection, document, 0)

# Get the file path from the user
file_path = input("Enter the path to the .txt file: ")

# Process the uploaded file
process_uploaded_file(file_path, collection)

# Example prompt
prompt = str(input("What is you're question? -> "))

# Generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
query_embedding = response["embedding"]

# Query the collection
results = collection.query(query_embeddings=[query_embedding], n_results=1)

# Check if any documents were returned
if results['documents']:
    data = results['documents'][0][0]

    # Generate a response combining the prompt and data
    output = ollama.generate(model="llama3-chatqa", prompt=f"Using this data: {data}. Respond to this prompt: {prompt} Also if you don't know the information. just say I don't know and don't make something up.")
    print(output['response'])
else:
    print("No relevant documents found in the collection.")

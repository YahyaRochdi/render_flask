import openai
import pinecone
from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI and Pinecone
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone_key = os.getenv("PINECONE_API_KEY")
openaiapi=os.getenv("OPENAI_API_KEY")

# configure client
pc = Pinecone(api_key=pinecone_key)

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)

# Initialize the embeddings model
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Connect to your Pinecone index
index_name = "metadata-book-by-page-3large"
index = pc.Index(index_name)
# def vectorize_query(text):
#     # Generate embedding for the query
#     embedding = embed_model.embed_documents([text]) # Assuming the embed method returns a list of embeddings
#     return embedding[0]
text_field = "text"

vectorstore = PineconeVectorStore(
    index, embed_model, text_field
)

import openai
import os
from openai import OpenAI




client = OpenAI(api_key=openaiapi)

system_prompt=""" You are an expert tutor in Project Management, 
with the PMBOK Guide 6th Edition as your primary reference tool. 
When a user asks a question or send a message, carefully evaluate whether it relates directly to the contents of the PMBOK Guide.
If the question is relevant, draw upon the specific knowledge from the guide to formulate your answer, ensuring accuracy and comprehensiveness.
Cite the relevant page numbers from the PMBOK Guide to enhance the credibility and usefulness of your responses.
Remember If a question falls outside the scope of the PMBOK Guide (pages), and answer as if the  PMBOK pages do not exist and provide a general answer.
"""
messages = [{"role": "system", "content": system_prompt}]

def get_response(user_input):
    # Take user input
    global messages
    print("User:" , user_input)
          
    

    augmented_input = user_input+" { PMBOK Guide : "+ str(vectorstore.similarity_search(user_input,k=5))+"} "
    user_dict = {"role": "user", "content": user_input}
    user_dict_augmented = {"role": "user", "content": augmented_input}
    messages_not_aug=messages.copy()
    messages_not_aug.append(user_dict)
    messages.append(user_dict_augmented)
    
    # Create the API request
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=4000
    )
    
    messages=messages_not_aug
    # Convert the assistant's message to a dict and append to the memory
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)
    
  # Print the assistant's response
    
    return response.choices[0].message.content
    
    # print("the message that was passed to the model for the last response is:" , messages_2)
    # print("the message that was recorded in history after the answer is  ",messages)
#########################################################################################################""
from flask import Flask, request, jsonify, render_template
import os

# Your existing chatbot code here

app = Flask(__name__)

@app.route('/')
def index():
    # Render the index.html page
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
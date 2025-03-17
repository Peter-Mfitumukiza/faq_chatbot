import utils.session
import openai
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environmental variables from .env file
load_dotenv()

# api_key
openai.api_key=os.getenv("OPENAI_API_KEY")

# initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "faq-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Load the data
with open('faq_dataset.json', "r", encoding="utf-8") as file:
    faq_data = json.load(file)

def store_faq_embeddings(): 
    for entry in faq_data:
        faq_id = entry['id']
        question = entry['question']
        answer = entry['answer']

        # Generate embeddings
        embedding_response = openai.embeddings.create(input=question, model= "text-embedding-ada-002")

         # Access the embedding using the object's attributes
        embedding = embedding_response.data[0].embedding

        # Store the embeddings in Pinecone 
        index.upsert(vectors=[(faq_id, embedding, {"question": question, "answer": answer})]);

        print("✅ FAQ embeddings stored in Pinecone!")


def get_answer(user_query):
    # Generate embbedding for user query
    query_embedding = openai.embeddings.create(input=user_query, model= "text-embedding-ada-002").data[0].embedding

    # Query Pinecone for closest match
    search_results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    # Extract the best match
    if search_results["matches"]:
        best_match = search_results["matches"][0]["metadata"]
        
        retrieved_answer = best_match["answer"]

        # Prepare a conversation that includes the user query and the retrieved context
        messages = [
            {"role": "system", "content": "You are a highly qualified and experienced food delivery app customer support expert. You are the best at answering customers queries."},
            {"role": "user", "content": (
                f"User question: {user_query}\n\n"
                f"Context: {retrieved_answer}\n\n"
                "Please provide a concise, helpful, and relevant answer and a step by step guide where applicable."
            )}
        ]
        
        # Generate the answer using the Chat Completion API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if available
            messages=messages,
            response_format={
                "type": "text"
            },
            temperature=0,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
            
        answer_text = response.choices[0].message.content
        
        return answer_text
    
    else:
        return "Sorry, I don't have an answer to that question."


if __name__ == "__main__":
    # store_faq_embeddings()
    user_question = input("Hi! How can I help you? ")
    answer = get_answer(user_question)
    print("Bot:", answer)

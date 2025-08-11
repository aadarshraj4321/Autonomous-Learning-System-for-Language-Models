import gradio as gr
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



DB_CHROMA_PATH = "db_chroma/"
MODEL_NAME = "EleutherAI/gpt-neo-125M"


print("Embedding model loading")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



print("Vector Database is loading")
db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)


print("LLM (GPT-Neo) loading")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)




def generate_response(user_prompt, history):
    print(f"New Question Come: {user_prompt}")

    relevant_docs = db.similarity_search(user_prompt, k=2)
    print(f"Relevant documents Find: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        print(f"Doc {i+1}: {doc.page_content[:100]}...")

    if relevant_docs:
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt_template = f"""
        Use the following context to answer the user's question. If the answer is not in the context, say "I do not have information about that."

        Context:
        ---
        {context}
        ---

        User's Question: {user_prompt}
        Answer:"""
    else:
        prompt_template = f"""
        Answer the following question.

        User's Question: {user_prompt}
        Answer:"""

    print("\nFinal prompt that sended to model")
    print(prompt_template)
    print("---------------------------------------\n")

    global pipe
    try:
        pipe
    except NameError:
        print("Text generation pipeline creating...")
        pipe = pipeline(
            'text-generation', 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=100,
            device=-1
        )

    generated_outputs = pipe(prompt_template)
    print(f"\nModel Raw Output")
    print(generated_outputs)
    print("----------------------------\n")

    if generated_outputs and len(generated_outputs) > 0:
        raw_generated_text = generated_outputs[0]['generated_text']
        final_response = raw_generated_text.replace(prompt_template, "").strip()
    else:
        final_response = "Sorry, I could not generate a response."

    print(f"Final Answer Sended")
    print(final_response)
    print("--------------------------------\n")
    
    return final_response




print("Creating Gradio UI")
demo = gr.ChatInterface(fn=generate_response, title="Autonomous Learning System for Language Model", 
                        description="You can ask about this project. the answer come inside the knowledge_base folder")



if __name__ == "__main__":
    print("Starting the application")
    demo.launch()
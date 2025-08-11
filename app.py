import gradio as gr
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from db_utils import log_feedback_to_db

# --- Constants ---
DB_CHROMA_PATH = 'db_chroma/'
MODEL_NAME = "EleutherAI/gpt-neo-125M"
MODEL_VERSION = "gpt-neo-125m-base-v1"

# --- Model Loading ---
@torch.no_grad()
def load_models():
    print("Models loading...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=150, device=-1)
    print("Models successfully loaded.")
    return embeddings, db, pipe

embeddings, db, pipe = load_models()

# --- Core Logic Functions ---
def format_chat_history(history):
    formatted_history = ""
    if history:
        for user_msg, ai_msg in history:
            if user_msg: formatted_history += f"User: {user_msg}\n"
            if ai_msg: formatted_history += f"AI: {ai_msg}\n"
    return formatted_history

def generate_response(user_prompt, history):
    chat_history_str = format_chat_history(history)
    relevant_docs = db.similarity_search(user_prompt, k=2)
    context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant context found."

    prompt_template = f"""You are a helpful AI assistant. Use the conversation history and the provided context to answer the user's question.

Conversation History:
{chat_history_str}

Context:
---
{context}
---

User's Question: {user_prompt}
Answer:"""

    generated_outputs = pipe(prompt_template)
    raw_generated_text = generated_outputs[0]['generated_text']
    final_response = raw_generated_text.replace(prompt_template, "").strip()

    state_data = {
        "prompt": user_prompt,
        "response": final_response
    }
    
    return final_response, state_data

def handle_feedback(feedback_type, state_data):
    if state_data and "prompt" in state_data:
        prompt = state_data["prompt"]
        response = state_data["response"]
        log_feedback_to_db(MODEL_VERSION, prompt, response, feedback_type)
        return gr.Info(f"Thank you for your '{feedback_type}' feedback!")
    return gr.Warning("Could not submit feedback. Please try again.")

def ui_controller(user_prompt, history, state_data):
    history.append([user_prompt, None])
    yield history, state_data

    response, new_state_data = generate_response(user_prompt, history[:-1])
    
    history[-1][1] = response
    
    yield history, new_state_data


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Adaptive Language Model Pipeline")
    gr.Markdown("This chatbot uses RAG to answer questions about its own project and learns from your feedback.")

    chatbot = gr.Chatbot(label="Chat History", bubble_full_width=False)
    state = gr.State({})

    with gr.Row():
        txt_input = gr.Textbox(show_label=False, placeholder="Enter your prompt here...", container=False)

    txt_input.submit(
        ui_controller,
        inputs=[txt_input, chatbot, state],
        outputs=[chatbot, state]
    )
    txt_input.submit(lambda: "", [], [txt_input])

    with gr.Row():
        thumb_up = gr.Button("👍 Thumbs Up")
        thumb_down = gr.Button("👎 Thumbs Down")

    thumb_up.click(
        lambda s: handle_feedback("up", s),
        inputs=[state],
        outputs=None
    )
    thumb_down.click(
        lambda s: handle_feedback("down", s),
        inputs=[state],
        outputs=None
    )

if __name__ == "__main__":
    demo.launch()
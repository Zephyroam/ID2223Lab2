import os
import gradio as gr
from huggingface_hub import InferenceClient

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

if not VLLM_BASE_URL:
    raise ValueError("Missing env var: VLLM_BASE_URL")
if not VLLM_API_KEY:
    raise ValueError("Missing env var: VLLM_API_KEY")

model2port = {
    "llama-3.2-1b-instruct-unsloth-bnb-16bit-FineTome-r32": 8000,
    "llama-3.2-3b-instruct-unsloth-bnb-16bit-FineTome-r32": 8001,
    "qwen-2.5-3b-instruct-unsloth-bnb-16bit-FineTome-r32": 8002,
}

def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    model_name,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """
    client = InferenceClient(token=VLLM_API_KEY, model=f"{VLLM_BASE_URL}:{model2port[model_name]}")

    model_name = f"Zephyroam/{model_name}"

    messages = [{"role": "system", "content": system_message}]

    messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        model=model_name,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Dropdown(
            label="Model name",
            choices=[
                "llama-3.2-1b-instruct-unsloth-bnb-16bit-FineTome-r32",
                "llama-3.2-3b-instruct-unsloth-bnb-16bit-FineTome-r32",
                "qwen-2.5-3b-instruct-unsloth-bnb-16bit-FineTome-r32",
            ],
            value="llama-3.2-3b-instruct-unsloth-bnb-16bit-FineTome-r32",
            allow_custom_value=False,
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()

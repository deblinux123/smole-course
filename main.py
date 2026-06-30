from transformers import pipeline
import torch





device = torch.device("cuda" if torch.cuda.is_available else "cpu")
modelName = "HuggingFaceTB/SmolLM3-3B"
# Pipeline Usage: Automated Chat Processing
pipe = pipeline("text-generation", modelName, device=device)

messages = [
    {"role":"system", "content":"you are helpfull and friendly chatbot that help programmers"},
    {"role":"user", "content":"can you tell me about python?"}
]

response = pipe(messages, max_new_tokens=128, temperature=0.7)

print(response[0]['generated_text'][-1])


# Advance pipe line usage

generation_config = {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

conversation = [
    {"role":"system", "content":"you are helpfull math tutor"},
    {"role":"user", "content":"can you help me with calculs?"}
]


response = pipe (conversation, **generation_config)
conversation = response[0]["generated_text"]
conversation.append({"role":"user", "content":"what is a derivative? can tell me in persian"})


response = pipe(conversation, **generation_config)

print("============================================")
print("Final conversation: \n")
print("============================================")

for message in response[0]['generated_text']:
    print(f"{message['role']} : {message['content']}")


# working with smolLM3 chat templates in code 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(modelName)

messages = [
    {"role":"system", "content": "you are helpfull assistant focused on technical topics and always answer the user in persian (farsi)."},
    {"role":"user", "content": "can you explain what a chat template is?"},
    {"role":"assistant", "content":"A chat tmeplate structures conversation between users and AI models by providing a consistent format that helps the model underestand diffrent rolse and maintain context."}
]

formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False, # return string instade of tokens
    add_generation_prompt=False # add prompt for next assistant response
)


print("=============================== FORMATTED CHAT IS: =================================\n")
print(formatted_chat)


print("=============================================")
print("PRINT WITH GENERATION PROMPT - FOR INFERANCE")
print("=============================================")


formatted_wit = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("==================================================================")
print("WITH GENERATION PROMPT:")
print("==================================================================")
print(formatted_wit)


print("\n\n\n")

standard_messages = [
    {"role": "user", "content": "What is 15 × 24?"},
    {"role": "assistant", "content": "15 × 24 = 360"}
]

# Thinking mode - show reasoning process
thinking_messages = [
    {"role": "user", "content": "What is 15 × 24?"},
    {"role": "assistant", "content": "<|thinking|>\nI need to multiply 15 by 24. Let me break this down:\n15 × 24 = 15 × (20 + 4) = (15 × 20) + (15 × 4) = 300 + 60 = 360\n</|thinking|>\n\n15 × 24 = 360"}
]


standard_formatted = tokenizer.apply_chat_template(standard_messages, tokenize=False)
thinking_formatted = tokenizer.apply_chat_template(thinking_messages, tokenize=False)


print("\n\nStandard mode:\n")
print(standard_formatted)

print("\n\nThinking mode:\n")
print(thinking_formatted)
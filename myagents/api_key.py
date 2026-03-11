import os

os.environ["ZHIPU_API_KEY"] = "sk_c5Y5ITLRO436DN6Y93BRQ733ukLavbpo_qViGETeZAA"
os.environ["ZHIPU_BASE_URL"] = "https://api.ppio.com/openai"
model_name = "zai-org/glm-4.7"  # used in code "client.chat.completions.create" and OpenAIChatCompletionsModel's parameter "model"
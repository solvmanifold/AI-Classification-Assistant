import os
import tempfile
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from unstructured.partition.pdf import partition_pdf

def process_pdf(pdf_file):
    temp_dir = tempfile.TemporaryDirectory()
    for file in pdf_file:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
            
    elements = partition_pdf(filename=temp_filepath,
                         infer_table_structure=True,
                         strategy='hi_res',
    )
    return [elem.metadata.text_as_html for elem in elements if elem.category == "Table"]

def prompt_template(classify_examples, user_query):
    
    return (f"""Human: You are highly qualified personnel who has been working as a analyst in secret service for government. You have been assigned to classify the documents based on the content. You have given a knowledge base which contains the information about\n
    Various government related documents.\n
    classify_examples : {{{classify_examples}}}\n
    user_query : {{{user_query}}}\n
    Follow the above `classify_examples` examples which was classified by you previously and use as reference to classify the `user_query` documents. Go through different examples for classification labels and respond with only \n
    classification label and rationale justification for your classification\n Assistant:""")

def llm_models():
    model_list=["openAI:GPT-4","llama2-70B-chat","openAI:GPT-4-turbo","mistral-7B-Instruct",
                "Anthropic:claude-v2","Anthropic:claude-2.1","Anthropic:claude-v1","gcp:gemini-pro",
                "Anthropic:claude-instant-1.2","mistral-large","databricks:dbrx-intruct"]
    
    model_params={"openAI:GPT-4":"https://clarifai.com/openai/chat-completion/models/GPT-4",
                  "llama2-70B-chat":"https://clarifai.com/meta/Llama-2/models/llama2-70b-chat",
                  "openAI:GPT-4-turbo":"https://clarifai.com/openai/chat-completion/models/gpt-4-turbo",
                  "mistral-7B-Instruct":"https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct",
                  "Anthropic:claude-v2":"https://clarifai.com/anthropic/completion/models/claude-v2",
                  "Anthropic:claude-2.1":"https://clarifai.com/anthropic/completion/models/claude-2_1",
                  "Anthropic:claude-v1":"https://clarifai.com/anthropic/completion/models/claude-v1",
                  "gcp:gemini-pro":"https://clarifai.com/gcp/generate/models/gemini-pro",
                  "Anthropic:claude-instant-1.2":"https://clarifai.com/anthropic/completion/models/claude-instant-1_2",
                  "mistral-large":"https://clarifai.com/mistralai/completion/models/mistral-large",
                  "databricks:dbrx-intruct":"https://clarifai.com/databricks/drbx/models/dbrx-instruct",
                  }
    return model_list, model_params

    
                        
                        
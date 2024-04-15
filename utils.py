import os
import tempfile

def prompt_template(classify_examples, user_query):
    prompt = f"""Human: Your job is to use information from the following classification guide to classify the input text.

    classification guide : {{{classify_examples}}}


    Compate the input text to the first column of the classification guide table to find the most relevant row.
    Your response should be limited to:
    "Classification: {{level\}}//{{dissem control}}, Derivative: {{derivative}}, Reason: {{reason}} De-classify on: {{declass info}}

    Justification: item
    Remarks: remarks", where level, dissem control, derivative, reason, declass info, and item are copied verbatium from the row in the classification guide that most relates to the input text. Note that "item" is found in the first column of a table. Note that in many cases dissem control, derivative, declass info, is missing: in that case do not respond with that information. You must only use information contained in the classification guide.
    In cases where the information in classification guide is not sufficient to confidently classify the input text, you must respond with "Unable to determine classification using the provided classification guide."

    Real human input
    input text : {{{user_query}}}
    Assistant:
    """
    return prompt

def few_shot_examples(vectorDB, user_query):
    
  resp = vectorDB.similarity_search(user_query)
  return list(set([doc.page_content for doc in resp]))

def few_shot_promt_template(few_shot_examples,user_query):
    
  prompt=f"""Human: Your job is to use information from the following classification guide to classify the input text.\n
  The classification guide consists of few shot examples that you can refer to answer the user's query.\n
  The few shot example is made of list with each field in the list corresponds to specific column :\n
  schema : ['Item','Level','Derivative','Dissemination Control','Reason','Declassify','Remarks'] \n
  few_shot_examples : {few_shot_examples},

  Your response should be limited to:
  "Classification: {{level\}}//{{dissem control}}, Derivative: {{derivative}}, Reason: {{reason}} De-classify on: {{declass info}}

  Justification: item
  Remarks: remarks", where level, dissem control, derivative, reason, declass info, and item are copied verbatium from the row in the classification guide that most relates to the input text. Note that "item" is found in the first column of a table. Note that in many cases dissem control, derivative, declass info, is missing: in that case do not respond with that information. You must only use information contained in the classification guide.
  In cases where the information in classification guide is not sufficient to confidently classify the input text, you must respond with "Unable to determine classification using the provided classification guide."

  Real human input
  input text : {{{user_query}}}
  Assistant:

  """
  return prompt

def llm_models():
    model_list=["openAI:GPT-4","llama2-70B-chat","openAI:GPT-4-turbo","mistral-7B-Instruct",
                "Anthropic:claude-v2","Anthropic:claude-2.1","Anthropic:claude-v1","gcp:gemini-pro",
                "Anthropic:claude-instant-1.2","mistral-large","mistral-small",
                "mistral-medium","databricks:dbrx-intruct", "Upstage:solar-10 7b-instruct"]
    
    model_params={"openAI:GPT-4":"https://clarifai.com/openai/chat-completion/models/GPT-4",
                  "llama2-70B-chat":"https://clarifai.com/meta/Llama-2/models/llama2-70b-chat",
                  "openAI:GPT-4-turbo":"https://clarifai.com/openai/chat-completion/models/gpt-4-turbo",
                  "mistral-7B-Instruct":"https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct",
                  "Anthropic:claude-v2":"https://clarifai.com/anthropic/completion/models/claude-v2",
                  "Anthropic:claude-2.1":"https://clarifai.com/anthropic/completion/models/claude-2_1",
                  "Anthropic:claude-v1":"https://clarifai.com/anthropic/completion/models/claude-v1",
                  "Upstage:solar-10 7b-instruct":"https://clarifai.com/upstage/solar/models/solar-10_7b-instruct",
                  "gcp:gemini-pro":"https://clarifai.com/gcp/generate/models/gemini-pro",
                  "Anthropic:claude-instant-1.2":"https://clarifai.com/anthropic/completion/models/claude-instant-1_2",
                  "mistral-large":"https://clarifai.com/mistralai/completion/models/mistral-large",
                  "mistral-small":"https://clarifai.com/mistralai/completion/models/mistral-small",
                  "mistral-medium":"https://clarifai.com/mistralai/completion/models/mistral-medium",
                  "databricks:dbrx-intruct":"https://clarifai.com/databricks/drbx/models/dbrx-instruct",
                  }
    
    return model_list, model_params

    
                        
                        
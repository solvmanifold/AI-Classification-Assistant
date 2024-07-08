import os
import requests
from concurrent.futures import ThreadPoolExecutor

def prompt_template(classify_examples, user_query):
    prompt = f"""Human: Your job is to use information from the following classification guide to classify the input text.

    classification guide : {{{classify_examples}}}

    Compare the input text to the first column of the classification guide table to find the most relevant row.
    Your response should reproduce the corresponding item in the classification guide formated as below:

  Item: {{Item}}
  Level: {{Level}}
  Derivative: {{Derivative}}
  Dissemination Control: {{Dissemination Control}}
  Reason: {{Reason}}
  Declassify on: {{Declassify}}
  Remarks: {{Remarks}}
    
    Where Item, Level, Derivativew, Reason, Declassify, Remarks are copied verbatim from the row in the classification guide where Item most closely relates to the input text.
    Item is found in the first column of the table.
    You must only use information contained in the classification guide.
    In cases where the information in classification guide is not sufficient to confidently classify the input text, you must respond with "Unable to determine classification using the provided classification guide."
    You must limit your response to only the single row that most closely relates to the input text, do not respond with multiple rows.

  Example 1:
  input text : The MIP's overall funding approved by congress for FY23 is $4.3B.
  Assistant:
  Item: (U) The aggregate or "top line" amount of funds requested and approved by Congress for the DoD Military Intelligence Program (MIP) for fiscal years 2007 through 2014.
  Level: 
  Derivative: 
  Dissemination Control: 
  Reason: 
  Declassify on: 
  Remarks: (U) No other MIP budget figures or program details will be released, as they remain classified for national security reasons.

  Example 2:
  input text : The Office of the Director of National Intelligence (ODNI) utilizes a detailed psychological profiling tool during its recruitment process to identify candidates who possess specific traits that are crucial for sensitive intelligence roles. This tool includes advanced algorithms that analyze candidates' responses to tailor-made scenarios reflecting real-world intelligence challenges. 
  Item: (U) Specific information concerning ODNI staff's recruitment, assessment, selection, and evaluation of applicants that reveals information which would allow this process to be circumvented.
  Level: S
  Derivative: ODNI HRM S-14
  Dissemination Control: NOFORN
  Reason: 1.4(c)
  Declassify on: Current date + 25 years
  Remarks: None

  Real human input
  input text : {{{user_query}}}
  Assistant:
    """
    return prompt

def retrieve_examples_rag(vectorDB, user_query, pat):
  response = list(vectorDB.query(ranks=[{"text_raw": user_query}],
                                 filters=[{'input_types': ['text']}]))
  hits =[hit for data in response for hit in data.hits]
  executor = ThreadPoolExecutor(max_workers=10)

  def hit_to_document(hit, pat):
    h= {"authorization": f"Bearer {pat}"}
    request = requests.get(hit.input.data.text.url, headers=h)
    request.encoding = request.apparent_encoding
    requested_text = request.text
    return (requested_text, "{:.3f}".format(hit.score))

  # Iterate over hits and retrieve hit.score and text
  futures = [executor.submit(hit_to_document, hit, pat) for hit in hits]
  docs_and_scores = list(set([future.result() for future in futures]))

  return docs_and_scores

def rag_prompt_template(rag_examples,user_query):
    
  prompt=f"""Human: Your job is to use information from the following classification guide to classify the input text.\n

  The classification guide consists of few shot examples that you can refer to answer the user's query.\n
  The classification guide is a python list where each field in the list corresponds to the following specific column :\n
  schema : ['Item','Level','Derivative','Dissemination Control','Reason','Declassify','Remarks'] \n

  classification guide : {rag_examples},

  Your response should reproduce the corresponding entry in the classification guide formated as below:

  Item: {{Item}}
  Level: {{Level}}
  Derivative: {{Derivative}}
  Dissemination Control: {{Dissemination Control}}
  Reason: {{Reason}}
  Declassify on: {{Declassify}}
  Remarks: {{Remarks}}

  You must limit your response to only the single entry in the classification guide that most closely relates to the input text, do not respond with multiple entries, and do not introduce any information not present in the classification guide.
  In cases where the information in classification guide does not sufficiently correspond to the input text, you must respond with "Unable to determine classification using the provided classification guide."

  Example 1:
  input text : The MIP's overall funding approved by congress for FY23 is $4.3B.
  Assistant:
  Item: (U) The aggregate or "top line" amount of funds requested and approved by Congress for the DoD Military Intelligence Program (MIP) for fiscal years 2007 through 2014.
  Level: 
  Derivative: 
  Dissemination Control: 
  Reason: 
  Declassify on: 
  Remarks: (U) No other MIP budget figures or program details will be released, as they remain classified for national security reasons.

  Example 2:
  input text : The Office of the Director of National Intelligence (ODNI) utilizes a detailed psychological profiling tool during its recruitment process to identify candidates who possess specific traits that are crucial for sensitive intelligence roles. This tool includes advanced algorithms that analyze candidates' responses to tailor-made scenarios reflecting real-world intelligence challenges. 
  Item: (U) Specific information concerning ODNI staff's recruitment, assessment, selection, and evaluation of applicants that reveals information which would allow this process to be circumvented.
  Level: S
  Derivative: ODNI HRM S-14
  Dissemination Control: NOFORN
  Reason: 1.4(c)
  Declassify on: Current date + 25 years
  Remarks: None

  Real human input
  input text : {{{user_query}}}
  Assistant:

  """
  return prompt

def llm_models():
    
    model_params = {'ai21:Jurassic2-Grande': 'https://clarifai.com/ai21/complete/models/Jurassic2-Grande',
                    'ai21:Jurassic2-Grande-Instruct': 'https://clarifai.com/ai21/complete/models/Jurassic2-Grande-Instruct', 
                    'ai21:Jurassic2-Jumbo': 'https://clarifai.com/ai21/complete/models/Jurassic2-Jumbo', 
                    'ai21:Jurassic2-Jumbo-Instruct': 'https://clarifai.com/ai21/complete/models/Jurassic2-Jumbo-Instruct', 
                    'ai21:Jurassic2-Large': 'https://clarifai.com/ai21/complete/models/Jurassic2-Large',
                    "anthropic:claude-3-opus":"https://clarifai.com/anthropic/completion/models/claude-3-opus",
                    'claude-3_5-sonnet"':'https://clarifai.com/anthropic/completion/models/claude-3_5-sonnet',
                    'anthropic:claude-2.1': 'https://clarifai.com/anthropic/completion/models/claude-2_1', 
                    'anthropic:claude-instant': 'https://clarifai.com/anthropic/completion/models/claude-instant', 
                    'anthropic:claude-instant-1.2': 'https://clarifai.com/anthropic/completion/models/claude-instant-1_2',
                    'anthropic:claude-v1': 'https://clarifai.com/anthropic/completion/models/claude-v1', 
                    'anthropic:claude-v2': 'https://clarifai.com/anthropic/completion/models/claude-v2',
                    'bigcode:StarCoder': 'https://clarifai.com/bigcode/code/models/StarCoder', 
                    'cohere:cohere-generate-command': 'https://clarifai.com/cohere/generate/models/cohere-generate-command', 
                    'cohere:command-r-plus': 'https://clarifai.com/cohere/generate/models/command-r-plus', 
                    'databricks:dbrx-instruct': 'https://clarifai.com/databricks/drbx/models/dbrx-instruct',
                    'databricks:dolly-v2-12b': 'https://clarifai.com/databricks/Dolly-v2/models/dolly-v2-12b',
                    'deci:deciLM-7B-instruct': 'https://clarifai.com/deci/decilm/models/deciLM-7B-instruct', 
                    'fblgit:una-cybertron-7b-v2': 'https://clarifai.com/fblgit/una-cybertron/models/una-cybertron-7b-v2',
                    'gcp:code-bison': 'https://clarifai.com/gcp/generate/models/code-bison', 
                    'gcp:code-gecko': 'https://clarifai.com/gcp/generate/models/code-gecko',
                    'gcp:gemini-pro': 'https://clarifai.com/gcp/generate/models/gemini-pro',
                    'gcp:gemma-1.1-7b-it': 'https://clarifai.com/gcp/generate/models/gemma-1_1-7b-it',
                    'gcp:gemma-2b-it': 'https://clarifai.com/gcp/generate/models/gemma-2b-it',
                    'gcp:gemma-7b-it': 'https://clarifai.com/gcp/generate/models/gemma-7b-it', 
                    'gcp:text-bison': 'https://clarifai.com/gcp/generate/models/text-bison', 
                    'huggingface-research:zephyr-7B-alpha': 'https://clarifai.com/huggingface-research/zephyr/models/zephyr-7B-alpha', 
                    'meta:Llama-3-8B-Instruct': 'https://clarifai.com/meta/Llama-3/models/Llama-3-8B-Instruct', 
                    'meta:codeLlama-70b-Instruct': 'https://clarifai.com/meta/Llama-2/models/codeLlama-70b-Instruct',
                    'meta:codeLlama-70b-Python': 'https://clarifai.com/meta/Llama-2/models/codeLlama-70b-Python',
                    'meta:llama2-13b-chat': 'https://clarifai.com/meta/Llama-2/models/llama2-13b-chat', 
                    'meta:llama2-70b-chat': 'https://clarifai.com/meta/Llama-2/models/llama2-70b-chat', 
                    'meta:llama2-7b-chat': 'https://clarifai.com/meta/Llama-2/models/llama2-7b-chat', 
                    'meta:llama2-7b-chat-vllm': 'https://clarifai.com/meta/Llama-2/models/llama2-7b-chat-vllm',
                    'meta:llamaGuard-7b': 'https://clarifai.com/meta/Llama-2/models/llamaGuard-7b',
                    'microsoft:phi-1.5': 'https://clarifai.com/microsoft/text-generation/models/phi-1_5',
                    'microsoft:phi-2': 'https://clarifai.com/microsoft/text-generation/models/phi-2', 
                    'mistralai:mistral-7B-Instruct': 'https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct',
                    'mistralai:mistral-7B-OpenOrca': 'https://clarifai.com/mistralai/completion/models/mistral-7B-OpenOrca',
                    'mistralai:mistral-large': 'https://clarifai.com/mistralai/completion/models/mistral-large', 
                    'mistralai:mistral-medium': 'https://clarifai.com/mistralai/completion/models/mistral-medium', 
                    'mistralai:mistral-small': 'https://clarifai.com/mistralai/completion/models/mistral-small',
                    'mistralai:mixtral-8x22B': 'https://clarifai.com/mistralai/completion/models/mixtral-8x22B', 
                    'mistralai:mixtral-8x7B-Instruct-v0.1': 'https://clarifai.com/mistralai/completion/models/mixtral-8x7B-Instruct-v0_1', 
                    'mistralai:openHermes-2-mistral-7B': 'https://clarifai.com/mistralai/completion/models/openHermes-2-mistral-7B', 
                    'mosaicml:mpt-7b-instruct': 'https://clarifai.com/mosaicml/mpt/models/mpt-7b-instruct',
                    'nousresearch:genstruct-7b': 'https://clarifai.com/nousresearch/instruction-generation/models/genstruct-7b',
                    'openai:GPT-3.5-turbo': 'https://clarifai.com/openai/chat-completion/models/GPT-3_5-turbo', 
                    'openai:GPT-4': 'https://clarifai.com/openai/chat-completion/models/GPT-4', 
                    'openai:gpt-3.5-turbo-instruct': 'https://clarifai.com/openai/completion/models/gpt-3_5-turbo-instruct', 
                    'openai:gpt-4-turbo': 'https://clarifai.com/openai/chat-completion/models/gpt-4-turbo', 
                    'openchat:openchat-3.5-1210': 'https://clarifai.com/openchat/openchat/models/openchat-3_5-1210', 
                    'salesforce:xgen-7b-8k-instruct': 'https://clarifai.com/salesforce/xgen/models/xgen-7b-8k-instruct', 
                    'tiiuae:falcon-40b-instruct': 'https://clarifai.com/tiiuae/falcon/models/falcon-40b-instruct', 
                    'togethercomputer:RedPajama-INCITE-7B-Chat': 'https://clarifai.com/togethercomputer/RedPajama/models/RedPajama-INCITE-7B-Chat', 
                    'togethercomputer:stripedHyena-Nous-7B': 'https://clarifai.com/togethercomputer/stripedHyena/models/stripedHyena-Nous-7B', 
                    'upstage:solar-10.7b-instruct': 'https://clarifai.com/upstage/solar/models/solar-10_7b-instruct', 
                    'wizardlm:wizardCoder-15B': 'https://clarifai.com/wizardlm/generate/models/wizardCoder-15B', 
                    'wizardlm:wizardCoder-Python-34B': 'https://clarifai.com/wizardlm/generate/models/wizardCoder-Python-34B', 
                    'wizardlm:wizardLM-13B': 'https://clarifai.com/wizardlm/generate/models/wizardLM-13B', 
                    'wizardlm:wizardLM-70B': 'https://clarifai.com/wizardlm/generate/models/wizardLM-70B'}
    
    return model_params

    
                        
def zero_shot_contents():
    return """ 
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<body>
<table cellspacing="0" cellpadding="0" class="t>
  <tbody>
    <tr>
      <th>Item</th>
      <th>Level</th>
      <th>Derivative</th>
      <th>Dissemination Control</th>
      <th>Reason</th>
      <th>Declassify on</th>
      <th>Remarks</th>
    </tr>
    <tr>
      <td>(U) The aggregate or "top line" amount of funds requested and approved by Congress for the DoD Military Intelligence Program (MIP) for fiscal years 2007 through 2014.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>(U) No other MIP budget figures or program details will be released, as they remain classified for national security reasons.</td>
    </tr>
    <tr>
      <td>(U) Budget or financial information directly tied to the NIP, below the aggregate or top line amount for all fiscal years.</td>
      <td>S</td>
      <td>ODNIFIN S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) Minimum SECRET//NOFORN. May be compartmented based on intelligence functional manager/OCA decision. Refer to appropriate intelligence program classification guide.</td>
    </tr>
    <tr>
      <td>(U) ODNI financial information inclusive of budget, personnel, expenditure, and appropriations data, to include related guidance, procedures, agreements, or disbursement techniques associated with ODNI administrative operations or projects/activities below the Directorate or Mission Manager level.</td>
      <td>U</td>
      <td></td>
      <td>FOUO</td>
      <td></td>
      <td></td>
      <td>(U) Unclassified, so long as the release o f such information will not lead to knowledge ofor insight into an intelligence community mission, sensitive program, target, vulnerability, capability, or intelligence source and method or budget information for any IC component - including the ODNI.</span><br></p>(U) Example: Overall budget of administrative functions for ODNl/HR, CHCO or PAO etc., is FOUO.</td>
    </tr>
    <tr>
      <td>(U) Specific information regarding ODNI or IC agency Directorate or Mission Manager-level fiscal matters inclusive of budget, expenditure, funding, and appropriations data, to include related guidance, procedures, agreements, or disbursement techniques, the release of which would provide the level of effort committed or insight into capabilities or intelligence sources or methods.</td>
      <td>S</td>
      <td>ODNI FIN-S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Detailed information regarding ODNI and IC fiscal matters inclusive of budget, manpower, expenditures, funding, and appropriations data, to include related guidance, procedures, agreements, vulnerabilities, or disbursement techniques related to the National Intelligence Program (NIP), or the financial condition and resources of the IC as a whole.</td>
      <td>TS</td>
      <td>ODNI FIN-T-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) A higher classification or additional markings may be required for compartmented information or programs. Consult with appropriate Program Compartment Guide or NCIX for compartmented info.</td>
    </tr>
    <tr>
      <td>(U) The fact that ODNI has an active recruitment, assessment, selection, and evaluation process to hire ODNI staff.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The fact that an overt employee (including their name) works for the ODNI as a staff person, detailee, or contractor.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) General information regarding ODNI staff recruitment, assessment, selection, and evaluation process.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Specific information concerning ODNI staff's recruitment, assessment, selection, and evaluation of applicants that reveals information which would allow this process to be circumvented.</td>
      <td>S</td>
      <td>ODNI HRM S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) General information concerning ODNI workforce, mission areas and administrative functions that do not describe the workforce structure in any detail or include names, titles, assignments or locations.</td>
      <td>U</td>
      <td></td>
      <td>FOUO</td>
      <td></td>
      <td></td>
      <td>(U) Does not include:- Specific number of staff, government and contractors in the operational areas (i.e., !ARPA, DDII, NIC, NCTC, NCPC, ONCIX, NIEMA, NIM staffs, and the IC CIO), implying priorities and scope, providing insight into sensitive aspects of operational or sensitive missions areas.- Resource totals.</span><br></p>(U) When in doubt, contact DNI-Classification for uidance.</td>
    </tr>
    <tr>
      <td>(U) Total number of ODNI or IC staff employees authorized or assigned.</td>
      <td>S</td>
      <td>ODNI HRM S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>U) Includes organizational charts where resource numbers can be deduced for the o erational elements within the ODNI.</td>
    </tr>
    <tr>
      <td>(U) Specific information concerning the existing or planned ODNI (or IC) workforce below the Mission Manager-level that describes the workforce structure including names, resource numbers, assignments or locations, which identifies an organization size and capability, or identifies the resources dedicated to an intelligence objective or geographical area.</td>
      <td>S</td>
      <td>ODNI HRM S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) Classified (notional) example:- ONCIX is comprised of 19 staffand 798 contractors working CI and Security related issues.- NCTC is authorized 500 staff billets, and 400 contractors to execute CT functions on behalf of the USG.- DDI I has 122 employees assigned.</span><br></p>(U) When in doubt, contact DNl-Classification for uidance.</td>
    </tr>
    <tr>
      <td>(U) Resource numbers (staff and/or contractor) covering non-operational ODNI elements, such as P&amp;S, PE.</td>
      <td>U</td>
      <td></td>
      <td>FOUO</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The fact that ODNI Headquarters is located within the Liberty Crossing Compound in the Tyson's Corner Area of Virginia.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>(U) Including the physical address of ODNI.</td>
    </tr>
    <tr>
      <td>(U) The fact that ODNI operates in facilities other than Liberty Crossing.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The name and abbreviation of a specific overt ODNI location in the Washington Metropolitan Area.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>(U) May be FOUO do to aggregation.</td>
    </tr>
    <tr>
      <td>(U) The association of the ODNI with a covert location.</td>
      <td>C</td>
      <td>ODNI LOC C-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) Refer to specific (covert) agency classification guidance for additional details.</td>
    </tr>
    <tr>
      <td>(U) The names and abbreviations of ODNI locations in the Washington Metropolitan Area, both overt and covert.</td>
      <td>S</td>
      <td>ODNI LOC S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>25X1 + 50 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Individually unclassified or controlled unclassified data items that in the compilation or aggregation would provide insight into ODNI's or IC's organization, functions, staffing, activities, capabilities, vulnerabilities, or intelligence sources or methods.</td>
      <td>C</td>
      <td>ODNI MOS C-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) May be REL TO USA, FVEY depending on the issue. Contact ODNI Partner Engagement (ODNl/PE) for guidance.</span><br></p>(U) Example I. Work products must be examined from an outsider's perspective to ensure adversaries cannot derive classified information from various unclassified sources. Unclassified "puzzle pieces" can sometimes be assembled to form a classified "picture" or unintentionallyÂ<span class="Apple-converted-space">  </span>paint a red X on a weak spot in your ability to perform or maintain your mission at acceptable levels.</td>
    </tr>
    <tr>
      <td>(U) Fact/existence of the Office of the Director of National Intelligence (ODNI).</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Fact that the Director of National Intelligence (DNI) serves as the head of the Intelligence Community (IC).</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The ODNI senior position titles, acronyms and abbreviations.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Fact that DNI serves as principal advisor to the President; the National Security Council, and Homeland Security Council for intelligence matters related to the national security.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Fact that DNI directs the implementation of the US government's National Intelligence Program (NIP) and special activities as directed by the President.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) ODNI organization charts that provide the existing or planned structure, positions and names assigned at the DNI, PDDNI, CMO, and/or DDNI/II level and one level below.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>(U) Includes Chiefs ofStaff, Executive-level Offices, Directorates, Centers, Administration, and Mission Managers.</td>
    </tr>
    <tr>
      <td>(U) Organization charts that provide specific details on the existing or planned structure of ODNI administrative, support or non-operational areas which include names, resource numbers, assignments or locations.</td>
      <td>S</td>
      <td></td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date plus 25 years</td>
      <td>(U) Includes: ARPA, DDll, NIC, NCTC, NCPC, ONCIX, NIEMA, NIM staffs, and the IC CIO,etc.(U) When in doubt, contact DNI-Classification for guidance.</td>
    </tr>
    <tr>
      <td>(U) Fact that ODNI employs CIA, FBI, NSA, DHS, DoS or other USG agency personnel (permanent staff or detailees) within the LX Compound, with no other details.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) General information regarding an overt contract that does not provide insight into capabilities, vulnerabilities, or intelligence sources or methods.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>(U) May require PROPIN or to be Contract Sensitive.</td>
    </tr>
    <tr>
      <td>(U) General information on analytic methodologies for sensitive data sets.</td>
      <td>C</td>
      <td>ODNI ANA C-14</td>
      <td>REL TO USA, FVEY</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>U) Does not include specifically authorized documents produced by NCTC or the NIC as pan of USG specific analyses or assessments which are unclassified for FOUO.</td>
    </tr>
    <tr>
      <td>(U) Intelligence analysis that provides general information regarding sensitive or classified collection systems or data sets, or intelligence sources and methods.</td>
      <td>C</td>
      <td>ODNI ANA C-14</td>
      <td></span><br></p>REL TO USA, FVEY</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) May contain companmented information. Refer to appropriate Program Security Officer I Program Security Guide for additional information.</td>
    </tr>
    <tr>
      <td>(U) Intelligence analysis that provides specific information regarding sensitive or classified collection systems or data sets, or intelligence sources and methods.</td>
      <td>S</td>
      <td>ODNI ANA C-14</td>
      <td></span><br></p>REL TO USA, FVEY</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Intelligence analysis that provides specific information regarding sensitive or classified collection systems or data sets, or intelligence sources and methods which if revealed, would nullify or measurably reduce their effectiveness.</td>
      <td>TS</td>
      <td>ODNI ANA C-14</td>
      <td></span><br></p>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td>(U) Contact ODNI Panner Engagement (ODNl/PE) for foreign disclosure guidance.</td>
    </tr>
    <tr>
      <td>(U) Information describing or derived from a US or foreign collection system, program, requirement, or R&amp;D effort where disclosure would reveal general US or foreign collection capability, or interest.</td>
      <td>S</td>
      <td>ODNI COL S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Information describing or derived from a US or foreign collection system, program, requirement, or R&amp;D effort where disclosure would hinder US collection.</td>
      <td>S</td>
      <td>ODNI COL S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Information describing or derived from a US or foreign collection system, program, requirement, or R&amp;D effort where disclosure would lessen US intelligence or collection advantage.</td>
      <td>S</td>
      <td>ODNI COL S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Information describing or derived from a US or foreign collection system, program, requirement, or R&amp;D effort where disclosure would reveal specific US or foreign collection capability, interest, or vulnerability.</td>
      <td>TS</td>
      <td>ODNI COL S-14</td>
      <td>NOFORN</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The fact that NCTC uses FISA-acquired and FISA-derived information in its analysis and in its intelligence products.</td>
      <td>U</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) Information obtained by, or derived from, an investigative technique requiring a FISA Court order or other FISA authorized collection ("FISA Information").</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(S//NF) Use of FISA-acquired and FISA-derived information when connected to a particular target or analytic judgment in intelligence products or analysis.</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(S//NF) The fact of NCTC's access to and use of raw, unminimized FISA-acquired information.</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(S//NF) The process and procedures of making raw, unminimized FISA-acquired information available to NCTC personnel.</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(S//NF) NCTC's Standard Minimization Procedures.</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(S//NF) The policies, process, and procedures for implementing NCTC's Standard Minimization Procedures.</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(S//NF) The fact that NCTC employees are required to sign a Letter of Consent to View and Work with Objectionable Material in order to access unminimized FISA acquired information.</td>
      <td>S</td>
      <td>ODNI FISA S-14</td>
      <td>NOFORN//FISA</td>
      <td>1.4(c)</td>
      <td>Current date + 25 years</td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The fact of the existence of the FISA Source Registry.</td>
      <td>U</td>
      <td></td>
      <td>FOUO</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>(U) The fact that all products that contain FISA-derived information must include the NCTC FISA caveat.</td>
      <td>U</td>
      <td></td>
      <td>FOUO</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</body>
</html>



"""              

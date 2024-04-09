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

def prompt_template():
    prompt="""You are highly qualified personnel who has been working as a analyst in secret service for government. You have been assigned to classify the documents based on the content. You have been given a document which contains the information about\n
    Various government related documents.\n
    {classify_examples} : {{{classification_examples}}}\n
    {user_query} : {{{question}}}\n
    Follow the above {classify_examples} examples to classify the {user_query} documents. These have been classified by you previously. Go through different classification labels and respond with only \n
    classification label and rational reasoning for the classification\n"""
    
    return PromptTemplate(
    input_variables=["classification_examples", "question"], template=prompt)

def few_shot_template(examples):
    PT= prompt_template()
    return FewShotPromptTemplate(
    examples=examples,
    example_prompt=PT,
    input_variables=["question"],
)


    
                        
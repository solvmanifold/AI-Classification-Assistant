# Module Classifier

Module Classifier is a Streamlit web application that utilizes LLM (Large Language Models) to classify documents provided by the user. The app uses the [ODNI classification guide](https://www.dni.gov/files/documents/FOIA/DF-2015-00044%20(Doc1).pdf) (office of Director of National Intelligence) to classify the document's category or class.

## ICL
The module has two modes where you can classify, ICl uses in-context learning method to pass all the classification sample guide into the prompt and classify the given doc from user.
![Screenshot 2024-04-23 at 5 09 10 PM](https://github.com/mogith-pn/module-classifier/assets/143642606/93fbdffa-b0d7-4027-934a-9c25a9b9b1a9)


## RAG
RAG mode utilises the retrieval of clarifai VectorDB for retrieving similar examples from the classification guide to classify the given document. When the context is too long and the model we can use RAG mode to effectively classify the docs.

![Screenshot 2024-04-23 at 5 07 49 PM](https://github.com/mogith-pn/module-classifier/assets/143642606/4ee8e115-a300-4dab-a502-29b73982e099)

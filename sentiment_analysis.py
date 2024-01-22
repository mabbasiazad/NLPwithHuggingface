'''
readme: https://huggingface.co/learn/nlp-course/en/chapter2/2?fw=pt
'''


CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
'''
Tokenizer
'''
from transformers import AutoTokenizer

checkpoint = CHECKPOINT #each model has its own tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#testing the tokenizer
raw_inputs = ["this sentence is going to be tokenized by tokenizer", 
              "this folder contains some code about NNP wiht hugging face"]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
print(inputs)

'''
model for generating hidden state 
which can be used for the task defined by the head layer
'''

from transformers import AutoModel

checkpoint =  CHECKPOINT
model = AutoModel.from_pretrained(checkpoint)

#testing the model
outputs = model(**inputs)
print(outputs.last_hidden_state.shape) #access by attribute you can also use access by key and index


'''
testing the model with classification head
'''
from transformers import AutoModelForSequenceClassification
checkpoint =  CHECKPOINT
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

#testing the model
outputs = model(**inputs)
print(outputs.logits) 

#post processing the output
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)  #using id2labe attribute of the model config


model.save_pretrained("./myModels") #save the model in the directory




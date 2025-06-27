# QnA-Learner
A simple RNN architecture that learns patterns of question structures and their respected answer.

I will use a basic RNN architecture to learn questions and their answers in the [dataset](https://github.com/UDAYVEER1312/QnA-Learner/blob/main/qa_dataset.csv) 

*Note that the model only learns and give preictinos from the dataset only*
# Dataset
[dataset](https://github.com/UDAYVEER1312/QnA-Learner/blob/main/qa_dataset.csv)  contains a set of 200 questions of History, Geography and Biology with equal portions.
NOTE : *The answers to some questions are integers instead of a string*

Example : When did American Civil war started? -> 1861 

# Model
Single layered [RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN) having [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) input of 60 dimensions and a hidden layer of 128 neurons.
using `torch`
```bash
import torch
from torch import nn
class simpleRNN(nn.Module):
    def __init__(self,embeddings,hidden_neurons,vocab_size):
        super().__init__()
        self.vocab = vocab_size
        self.embd = nn.Embedding(num_embeddings = self.vocab , embedding_dim = embedding)
        self.rnn = nn.RNN(input_size = embeddings , hidden_size =  hidden_neurons, batch_first=  True)
        self.out = nn.Linear(hidden,self.vocab)
    def forward(self,text):
        x = self.embd(text)
        hidden_states , final_hidden_state = self.rnn(x)
        return self.out(final_hidden_state.squeeze(0))
```

# Tokenization and Preprocessing
Let's say if dataset only consist of a single question : *Which is the tallest building?* -> *Burj Khalifa*
then the tokenized words would be `{which, is, the, tallest, building, burjkhalifa}`

Note that all words are in lowercase and `?` is removed

You can use `nltk` to tokenize texts but since the questions are simple I use a manual tokenization function
```bash
def tokenize(text,question = True):  
    if not question:
        return list(text.replace(' ','').split())
    text = text.lower()
    text = text.replace('?','')
    return text.split()
```
The argument `question` in `tokenize(text,question=True)` is initialized due to the varying word size of answers ,i.e, the answer `South Africa` should tokenized to `southafrica` since the model is contructed for sequence input to single output
Create Vocab
```bash
vocab = {'<unk>':0}   # 0 for unknown words
for q,a in zip(data['question'],data['answer']):
    tokenized_q = tokenize(q)
    tokenized_a = str(a)
    if isinstance(a,str):
     tokenized_a = tokenize(a,False)
    for word in tokenized_q+tokenized_a :
        if word not in vocab.keys():
            vocab[word] = len(vocab)
```
To convert text questions to a sequence of numbers we need a token to index function.

The below function handles the issue of tokenization of answers as well
```bash
def text_to_indices(text,vocab,question=True):
    t_to_id = []
    if not question:
        tokenized_text = str(text)
        tokenized_text = tokenize(text,False)
    else:
        tokenized
_text = tokenize(text)
    for t in tokenized_text:
        if t in vocab.keys(): 
            t_to_id.append(vocab[t])
        else:
            t_to_id.append(vocab['<unk>'])
    return t_to_id
```
# Custom dataset
Using `torch` and creating a custom dataset  
```bash
from torch.utils.data import Dataset, DataLoader
class Custom_Dataset(Dataset):
    def __init__(self,data,vocab):
        super().__init__()
        self.data = data
        self.vocab = vocab
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,index):
        text_to_id_question = text_to_indices(self.data['question'][index],self.vocab)
        text_to_id_answer = text_to_indices(self.data['answer'][index],self.vocab, False)
        return torch.tensor(text_to_id_question) , torch.tensor(text_to_id_answer)
```
For DataLoader and Training see the `ipynb` file [QnA_model.ipynb]()
# Predictions
What if we make some changes to a question and then try to predict the answer that the model has learned so far?

So, not only we want our model to learn relationship between a *question* and its *answer* but also relationships within a *question*. So do i need to make my model more complex...?

Well the answer is simply NO.

Since `RNN` is used the model finds those relationships as it is the primary task for such sequence architectures. So even a minute vocabulary change to our input will not affect the model's prediction.

*If the question is completely out of the trained dataset then the model won't be able to predict it's answer.*

For further info regarding the predictions I shared the code snippet for Prediction function below.
```bash
def predict(question,threshold = 0.1):
    tensor_input = torch.tensor(text_to_indices(question,vocab))
    pred_logits = rnn_model(tensor_input.unsqueeze(dim=0))
    pred_probs = nn.functional.softmax(pred_logits,dim=1)
    max_prob , prediction = torch.max(pred_probs,dim=1)
    print(f'probability : {max_prob.item()}')
    if max_prob.item() < threshold:
        return 'No idea...'
    return list(vocab.keys())[prediction]
predict("Organ that filters blood?")
```
---output---
```bash
probability : 0.9454696774482727
'Kidney'
```
Actual question : Which organ filters blood?

Asked question : Organ that filters blood?

*NOTE that a threshold is required while predicting. If the probability is lower than the threshold then the modle has encountered an out of syllabus question.* 

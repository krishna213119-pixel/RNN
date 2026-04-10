import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Page Config ---
st.set_page_config(page_title="RNN Q&A System", page_icon="🧠", layout="centered")

# --- Custom Styling ---
CUST_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap');
    
    /* Background and Overall Font */
    .stApp {
        background-color: #0d0d0d;
        color: #f0f0f0;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #ff3333;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #4d0000;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff1a1a;
        box-shadow: 0 0 8px rgba(255, 26, 26, 0.6);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #8b0000 0%, #4d0000 100%);
        color: #ffffff;
        border: 1px solid #ff1a1a;
        border-radius: 6px;
        padding: 10px 24px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a1a 0%, #aa0000 100%);
        border-color: #ff4d4d;
        box-shadow: 0 0 12px rgba(255, 26, 26, 0.8);
        transform: translateY(-2px);
    }
    
    /* Answer box */
    .answer-box {
        background-color: rgba(26, 0, 0, 0.6);
        border-left: 4px solid #ff1a1a;
        padding: 20px;
        border-radius: 4px;
        font-size: 20px;
        font-weight: 500;
        color: #ffe6e6;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.1);
    }
    
    hr {
        border-color: #330000;
    }
</style>
"""
st.markdown(CUST_CSS, unsafe_allow_html=True)


# --- Model Definitions ---
def tokenize(text):
    text = text.lower()
    text = text.replace('?','')
    text = text.replace("'","")
    return text.split()

def text_to_indices(text, vocab):
    indexed_text = []
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab['<UNK>'])
    return indexed_text

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, question):
        embedded_question = self.embedding(question)
        hidden, final = self.rnn(embedded_question)
        output = self.fc(final.squeeze(0))
        return output

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        numerical_question = text_to_indices(self.df.iloc[index]['question'], self.vocab)
        numerical_answer = text_to_indices(self.df.iloc[index]['answer'], self.vocab)
        return torch.tensor(numerical_question), torch.tensor(numerical_answer)


# --- App Logic & Loading ---
@st.cache_resource(show_spinner="Initializing Neural Network...")
def load_and_train_model():
    # Attempt to load dataset
    try:
        df = pd.read_csv('100_Unique_QA_Dataset.csv')
    except FileNotFoundError:
        return None, None, "Dataset '100_Unique_QA_Dataset.csv' not found. Please ensure it's in the same directory."

    # Build vocab
    vocab = {'<UNK>':0}
    def build_vocab(row):
        tokenized_question = tokenize(row['question'])
        tokenized_answer = tokenize(row['answer'])
        merged_tokens = tokenized_question + tokenized_answer
        for token in merged_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    df.apply(build_vocab, axis=1)

    # Prepare Dataset
    dataset = QADataset(df, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize Model
    model = SimpleRNN(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    epochs = 40  # Increased slightly for better fit, fast anyway
    for epoch in range(epochs):
        for question, answer in dataloader:
            optimizer.zero_grad()
            output = model(question)
            loss = criterion(output, answer[0])
            loss.backward()
            optimizer.step()

    model.eval() # Set to evaluation mode
    return model, vocab, None


# --- UI ---
st.title("🔴 RNN Q&A Engine")
st.markdown("Ask anything available in our trained corpus. Pure neural processing.")
st.markdown("---")

model, vocab, err_msg = load_and_train_model()

if err_msg:
    st.error(err_msg)
else:
    question_input = st.text_input("Enter your question:", placeholder="What is the capital of France?", key="q_input")
    
    if st.button("Predict Answer"):
        if question_input:
            with st.spinner("Processing..."):
                numerical_question = text_to_indices(question_input, vocab)
                if len(numerical_question) == 0:
                     st.warning("Please enter a valid question.")
                else:
                    question_tensor = torch.tensor(numerical_question).unsqueeze(0)
                    with torch.no_grad():
                        output = model(question_tensor)
                    
                    probs = torch.nn.functional.softmax(output, dim=1)
                    value, index = torch.max(probs, dim=1)
                    
                    threshold = 0.4
                    if value.item() < threshold:
                        answer_text = "I don't know"
                    else:
                        answer_text = list(vocab.keys())[index.item()]
                    
                    # Display Answer
                    st.markdown(f'<div class="answer-box">⚡ <b>Response:</b> {answer_text}</div>', unsafe_allow_html=True)
        else:
            st.warning("You must type a question first.")

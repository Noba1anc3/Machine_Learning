# Generative Deep Learning

- Music Writing
- Dialogue 
- Image Generation
- Speech Synthesis
- Molecular Design
- Script Design
- ...

## Text Generating from LSTM

- Text
- Music
- Drawing

### A Brief History of Generative Recurrent Network

- Until end of 2014, almost nobody know LSTM
- Get into main area since 2016
- Earliest - 1997
- First apply in music generation, and get a satisfying  - 2002
- Apply recurrent mix density network on generating human handwriting - 2013

### How to generate sequence data

General Method : Use former tokens input, train a RNN / CNN to predict next one (or more) token(s).  

#### Language Model

Given former tokens, any network able to modeling the probability of next token.  

Language model can capture the latent space of a language, aka, the statistic structure of a language.  

#### Sample

Given a language model, generate a new sequence.

### The Importance of Sample Strategy

#### Greedy Sampling

- Always select the character with max probability

- Generate a repetitive, predictable string
- Do not look like coherent language

#### Stochastic Sampling

- Introduce randomness during sampling
- If the next character is 'e''s probability is 0.3, then model will choose it with a probability of 30%

Whether minimum-entropy selection (greedy sampling) or maximum-entropy selection (completely random sampling) would not generate anything interesting.  

Sampling in the 'real' probability distribution (softmax) is a compromise.  

- Bigger Entropy : Creativity
- Smaller Entropy : Authentic

To control randomness during sampling, we introduce a parameter called 'softmax temperature' to indicate the entropy of sample probability distribution, aka, indicates how unexpected or predictable the next character you select.  
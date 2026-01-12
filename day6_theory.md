# Day 6: Transformers & Large Language Models

## Overview

Transformers have revolutionized not just NLP but all of AI. Understanding transformer architecture and LLMs is **critical** for Principal AI Architects because:
- LLMs are now central to most AI strategies
- System design questions frequently focus on LLM deployment
- Infrastructure requirements are unique and challenging
- RAG (Retrieval Augmented Generation) is becoming standard pattern
- Interview questions test both theoretical understanding and practical deployment

**Why This Day Matters:**
- Transformers power GPT, BERT, Claude, ChatGPT, and nearly all modern NLP
- Attention mechanism applies beyond NLP (Vision Transformers, protein folding)
- LLM deployment involves unique challenges (inference cost, latency, scaling)
- RAG bridges LLMs with enterprise data

---

## 1. Transformer Architecture - Deep Dive

### The Original Paper: "Attention Is All You Need" (2017)

**Revolutionary Claims:**
1. No recurrence needed (pure attention)
2. Fully parallelizable (unlike RNNs)
3. State-of-the-art results on translation
4. Foundation for modern NLP

### High-Level Architecture

```
INPUT: "The cat sat on the mat"

┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER                               │
│                                                              │
│  ┌────────────────────┐         ┌────────────────────┐     │
│  │     ENCODER        │         │     DECODER        │     │
│  │   (6 layers)       │────────▶│   (6 layers)       │     │
│  │                    │         │                    │     │
│  │ Self-Attention     │         │ Masked Attn        │     │
│  │ Feed-Forward       │         │ Cross-Attn         │     │
│  │ (repeat 6×)        │         │ Feed-Forward       │     │
│  └────────────────────┘         │ (repeat 6×)        │     │
│           │                     └────────────────────┘     │
│           │                              │                  │
│           └──── Encoder Output ──────────┘                  │
│                                          │                  │
└──────────────────────────────────────────┼──────────────────┘
                                           ↓
                                    OUTPUT PROBABILITIES
```

**Key Point:** Most modern models (BERT, GPT) use only encoder or only decoder, not both!

---

## 2. Self-Attention Mechanism - The Core Innovation

### What Problem Does Attention Solve?

**RNN Limitation:**
```
Sentence: "The cat that scared the dog that chased the mouse ran away"

RNN processes sequentially:
Step 1: Process "The" → h1
Step 2: Process "cat" using h1 → h2
...
Step 12: Process "ran" using h11 → h12

Problem: h12 must encode entire sentence
By step 12, "cat" information degraded (vanishing gradient)
```

**Attention Solution:**
```
Every word directly attends to every other word
"ran" can directly look at "cat" (no intermediate steps)
Path length = 1 (constant, regardless of distance)
No information degradation
```

### How Self-Attention Works

**Input:** Sequence of word embeddings
**Output:** Sequence of context-aware representations

**Three Components: Query, Key, Value**

```
For each word, create three vectors:
- Query (Q): "What am I looking for?"
- Key (K): "What do I offer?"
- Value (V): "What information do I carry?"

All created by linear transformations of word embedding:
Q = W_Q × embedding
K = W_K × embedding
V = W_V × embedding
```

**Step-by-Step Example:**

```
Sentence: "The cat sat"
Embeddings: e_the, e_cat, e_sat (each 512-dim)

Step 1: Create Q, K, V for each word
────────────────────────────────────
Word   Query        Key          Value
The    Q_the        K_the        V_the
cat    Q_cat        K_cat        V_cat
sat    Q_sat        K_sat        V_sat

(Each Q, K, V is 64-dim after projection)


Step 2: Calculate attention scores (for "cat")
────────────────────────────────────────────────
How relevant is each word to "cat"?

Score = Q_cat · K_word (dot product measures similarity)

Score(cat, The) = Q_cat · K_the = 20
Score(cat, cat) = Q_cat · K_cat = 95
Score(cat, sat) = Q_cat · K_sat = 60


Step 3: Scale scores
────────────────────────
Why: Prevents large dot products (numerical stability)

Scaled = Score / √d_k
d_k = dimension of key (64)

Scaled(cat, The) = 20 / 8 = 2.5
Scaled(cat, cat) = 95 / 8 = 11.875
Scaled(cat, sat) = 60 / 8 = 7.5


Step 4: Apply softmax (convert to probabilities)
─────────────────────────────────────────────────
Attention weights sum to 1

Weights = softmax([2.5, 11.875, 7.5])
        = [0.05, 0.70, 0.25]

"cat" attends:
- 70% to itself
- 25% to "sat" (related action)
- 5% to "The" (less relevant)


Step 5: Weighted sum of values
───────────────────────────────
Output for "cat":
output_cat = 0.05 × V_the + 0.70 × V_cat + 0.25 × V_sat

This is the new context-aware representation of "cat"
```

**Mathematical Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

QK^T: All query-key dot products (attention scores)
√d_k: Scaling factor
softmax: Convert to probabilities (row-wise)
· V: Weighted sum of values
```

**Matrix Form (Efficient Computation):**

```
Input: Matrix of embeddings [seq_len × d_model]

Q = X · W_Q  [seq_len × d_k]
K = X · W_K  [seq_len × d_k]
V = X · W_V  [seq_len × d_v]

Scores = Q · K^T  [seq_len × seq_len]
Attention = softmax(Scores / √d_k)  [seq_len × seq_len]
Output = Attention · V  [seq_len × d_v]

All words processed in parallel (efficient!)
```

### Why This Works

**1. Direct Connections:**
```
Every word can directly attend to every other word
No intermediate steps (unlike RNN)
Path length = 1 (constant)
No vanishing gradients for long-range dependencies
```

**2. Context-Aware Representations:**
```
"bank" in "river bank" attends to "river" → geography context
"bank" in "bank account" attends to "account" → finance context
Same word, different representations (contextual!)
```

**3. Learnable Weights:**
```
W_Q, W_K, W_V learned during training
Model learns what to attend to
Different layers learn different patterns
```

---

## 3. Multi-Head Attention

### Why Multiple Heads?

**Problem with Single Attention:**
```
One attention mechanism = one way of relating words
But language has multiple relationships:
- Syntactic (subject-verb agreement)
- Semantic (word meanings)
- Positional (nearby words)
- Coreference (pronouns referring to nouns)
```

**Solution: Multiple Attention Heads in Parallel**

```
Typical: 8-16 heads

Head 1: Syntactic patterns
  "The cat sat" → Head 1 attends cat→sat (subject-verb)

Head 2: Semantic relationships
  "The cat sat" → Head 2 attends cat→animal concepts

Head 3: Local context
  "The cat sat" → Head 3 attends to adjacent words

Head 4-8: Other patterns learned during training
```

### Multi-Head Attention Mechanism

```
┌─────────────────────────────────────────────────────┐
│         MULTI-HEAD ATTENTION                        │
│                                                      │
│  Input (d_model = 512)                              │
│         │                                            │
│         ├──────┬──────┬──────┬──────┬──────┬──────┐│
│         ↓      ↓      ↓      ↓      ↓      ↓      ││
│       Head1  Head2  Head3  Head4  Head5  ... Head8 ││
│       (64d)  (64d)  (64d)  (64d)  (64d)     (64d) ││
│         │      │      │      │      │        │     ││
│         └──────┴──────┴──────┴──────┴────────┘     ││
│                      │                              │
│              Concatenate (512d)                     │
│                      │                              │
│              Linear (512 → 512)                     │
│                      │                              │
│                   Output                            │
└─────────────────────────────────────────────────────┘

Each head:
- Has own W_Q, W_K, W_V matrices
- Operates on smaller dimension (512/8 = 64)
- Learns different attention patterns
- Outputs concatenated and linearly transformed
```

**Mathematical Formula:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

h = number of heads (typically 8)
Each head has dimension d_k = d_model / h
```

**Why This Design:**

```
Computation:
8 heads × 64 dimensions ≈ same cost as 1 head × 512 dimensions

Benefits:
- Multiple representational subspaces
- Richer feature learning
- Better performance empirically

Cost:
- Minimal (same total parameters)
```

---

## 4. Positional Encoding

### The Position Problem

**Attention is Position-Agnostic:**
```
"cat sat mat" has same attention as "mat sat cat"
Order information lost!

But order matters:
"dog bites man" ≠ "man bites dog"
```

**Solution: Add Positional Information to Embeddings**

### Positional Encoding Formula

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos: Position in sequence (0, 1, 2, ...)
i: Dimension index
d_model: Embedding dimension (512)

Even dimensions: sine
Odd dimensions: cosine
```

**Why This Formula:**

```
1. Unique encoding for each position
   PE(0) ≠ PE(1) ≠ PE(2) ≠ ...

2. Relative positions detectable
   Model can learn "word 3 positions after word 1"

3. Generalizes to unseen sequence lengths
   Formula defined for any position

4. Smooth variations
   Similar positions → Similar encodings
```

**Usage:**

```
Word embedding: [0.2, -0.4, 0.7, ...]  (512-dim)
Positional encoding: [0.1, 0.05, -0.3, ...]  (512-dim)

Final input: embedding + positional_encoding

Model now knows both:
- What the word is (embedding)
- Where it is (positional encoding)
```

**Learned vs Fixed Positional Encodings:**

```
Original Transformer: Fixed (sinusoidal)
BERT: Learned (trainable embeddings for each position)
GPT: Learned

Modern trend: Learned positional encodings
Often perform slightly better
```

---

## 5. Complete Transformer Layer

### Encoder Layer

```
┌─────────────────────────────────────────────┐
│           ENCODER LAYER                     │
│                                              │
│  Input (from previous layer or embeddings)  │
│              ↓                               │
│  ┌──────────────────────────────┐          │
│  │  Multi-Head Self-Attention   │          │
│  └──────────────────────────────┘          │
│              ↓                               │
│         Add & Norm  ←────────────┐         │
│              ↓                    │         │
│  ┌──────────────────────────────┐│         │
│  │  Feed-Forward Network        ││  Skip   │
│  │  (2 linear layers + ReLU)    ││  Connections │
│  └──────────────────────────────┘│         │
│              ↓                    │         │
│         Add & Norm  ←────────────┘         │
│              ↓                               │
│           Output                            │
└─────────────────────────────────────────────┘
```

**Components:**

**1. Multi-Head Self-Attention:**
```
All words attend to all words
Context-aware representations
```

**2. Add & Norm (Residual Connection + Layer Normalization):**
```
output = LayerNorm(x + Sublayer(x))

Residual (x +): Helps gradient flow (like ResNet)
LayerNorm: Stabilizes training (normalizes across features)
```

**3. Feed-Forward Network:**
```
FFN(x) = max(0, x·W1 + b1)·W2 + b2

Two linear transformations with ReLU
Applied to each position independently (no interaction between positions)
Expands then compresses: 512 → 2048 → 512
Adds non-linearity and capacity
```

**4. Residual Connections:**
```
Why: Enable training very deep networks (100+ layers)
How: Skip connections bypass transformations
Benefit: Gradients flow directly backward
```

### Decoder Layer

```
┌─────────────────────────────────────────────┐
│           DECODER LAYER                     │
│                                              │
│  Input (from previous layer)                │
│              ↓                               │
│  ┌──────────────────────────────┐          │
│  │  Masked Multi-Head           │          │
│  │  Self-Attention              │          │
│  │  (can't see future)          │          │
│  └──────────────────────────────┘          │
│              ↓                               │
│         Add & Norm  ←────────────┐         │
│              ↓                    │         │
│  ┌──────────────────────────────┐│         │
│  │  Multi-Head Cross-Attention  ││  Skip   │
│  │  (attend to encoder output)  ││  Connections │
│  └──────────────────────────────┘│         │
│              ↓                    │         │
│         Add & Norm  ←────────────┤         │
│              ↓                    │         │
│  ┌──────────────────────────────┐│         │
│  │  Feed-Forward Network        ││         │
│  └──────────────────────────────┘│         │
│              ↓                    │         │
│         Add & Norm  ←────────────┘         │
│              ↓                               │
│           Output                            │
└─────────────────────────────────────────────┘
```

**Additional Components:**

**1. Masked Self-Attention:**
```
During generation, can't see future words
Mask: Set future attention scores to -∞
After softmax: future words have 0 attention

Example (generating "The cat sat"):
When predicting "sat":
- Can attend to: "The", "cat"
- Cannot attend to: future words (not generated yet)
```

**2. Cross-Attention (Encoder-Decoder Attention):**
```
Decoder attends to encoder output

Query: From decoder (what we're generating)
Key, Value: From encoder (source sentence)

Example (translation):
Source: "I love cats"
Target: "J'aime les chats"

When generating "chats":
- Query: decoder state for "chats"
- Keys/Values: encoder representations of "I", "love", "cats"
- Attends strongly to "cats" (translating this word)
```

---

## 6. BERT vs GPT vs T5 - Model Families

### 6.1 BERT (Bidirectional Encoder)

**Architecture: Encoder-Only**

```
┌────────────────────────────────────┐
│            BERT                    │
│                                     │
│  Input: [CLS] Sentence [SEP]       │
│           ↓                         │
│  ┌─────────────────────┐           │
│  │  Encoder × 12/24    │           │
│  │  (Bidirectional)    │           │
│  └─────────────────────┘           │
│           ↓                         │
│  Contextual Embeddings              │
│                                     │
│  [CLS] → Classification             │
│  Tokens → Token Classification      │
└────────────────────────────────────┘
```

**Pre-training Tasks:**

**1. Masked Language Model (MLM):**
```
Original: "The cat sat on the mat"
Masked: "The [MASK] sat on the [MASK]"
Task: Predict masked words

15% of tokens masked:
- 80% replaced with [MASK]
- 10% replaced with random word
- 10% unchanged (prevents model from only learning masked positions)

Forces bidirectional understanding:
"The [MASK] sat" (left context) + "sat on the" (right context)
```

**2. Next Sentence Prediction (NSP):**
```
Given two sentences:
Sentence A: "I love cats"
Sentence B: "They are cute" (IsNext: 50%)
         or "Paris is beautiful" (NotNext: 50%)

Task: Binary classification (IsNext or NotNext)

Teaches: Sentence relationships, discourse understanding
Later removed in RoBERTa (didn't help much)
```

**Model Variants:**

```
BERT-Base:
- 12 layers, 768 hidden, 12 heads
- 110M parameters
- Good for most tasks

BERT-Large:
- 24 layers, 1024 hidden, 16 heads
- 340M parameters
- State-of-the-art (but slower)

RoBERTa (Robustly Optimized BERT):
- Same architecture
- Better pre-training (remove NSP, more data, longer training)
- Improved performance

DistilBERT:
- 6 layers (half of BERT)
- 66M parameters
- 97% performance, 60% faster
- Knowledge distillation
```

**Use Cases:**
```
✓ Classification (sentiment, topic, intent)
✓ Named Entity Recognition
✓ Question Answering (extractive)
✓ Any understanding task

✗ Text generation (not designed for this)
✗ Translation (encoder-only)
```

### 6.2 GPT (Generative Pre-trained Transformer)

**Architecture: Decoder-Only**

```
┌────────────────────────────────────┐
│            GPT                     │
│                                     │
│  Input: "The cat sat"              │
│           ↓                         │
│  ┌─────────────────────┐           │
│  │  Decoder × 12-96    │           │
│  │  (Causal/Masked)    │           │
│  │  Left-to-right only │           │
│  └─────────────────────┘           │
│           ↓                         │
│  Next Token Prediction              │
│  "on" (most likely)                 │
└────────────────────────────────────┘
```

**Pre-training Task: Language Modeling**

```
Given: "The cat sat on the"
Predict: "mat"

Given: "The cat sat on the mat in"
Predict: "the"

Autoregressive:
- Predict next word
- Use prediction as input for next step
- Trained on web text (billions of words)
```

**Causal (Masked) Attention:**

```
When processing "sat", can only attend to:
- "The"
- "cat"
- "sat" (itself)

Cannot attend to:
- "on" (future word)
- "the" (future)
- "mat" (future)

Enforced by attention mask:
Set future positions to -∞ before softmax
```

**Model Evolution:**

```
GPT-1 (2018):
- 12 layers, 768 hidden
- 117M parameters
- Trained on BooksCorpus
- Showed transfer learning works

GPT-2 (2019):
- 48 layers, 1600 hidden
- 1.5B parameters
- Zero-shot: Works without fine-tuning
- "Too dangerous to release" (later released)

GPT-3 (2020):
- 96 layers, 12288 hidden
- 175B parameters
- Few-shot learning (learns from examples in prompt)
- No fine-tuning needed!

GPT-4 (2023):
- Architecture unknown (estimated 1T+ parameters)
- Multimodal (text + images)
- Improved reasoning, reduced hallucinations
- Longer context (32K tokens)
```

**Use Cases:**
```
✓ Text generation (articles, stories, code)
✓ Dialogue systems
✓ Translation (via prompting)
✓ Summarization (via prompting)
✓ Few-shot learning (any task with examples)

✗ Classification (possible but inefficient)
✗ NER (possible but BERT better)
```

### 6.3 T5 (Text-to-Text Transfer Transformer)

**Architecture: Full Encoder-Decoder (like original Transformer)**

```
┌────────────────────────────────────┐
│              T5                    │
│                                     │
│  Input: "translate English to      │
│          German: I love cats"       │
│           ↓                         │
│  ┌─────────────────────┐           │
│  │  Encoder × 12/24    │           │
│  └─────────────────────┘           │
│           ↓                         │
│  ┌─────────────────────┐           │
│  │  Decoder × 12/24    │           │
│  └─────────────────────┘           │
│           ↓                         │
│  Output: "Ich liebe Katzen"        │
└────────────────────────────────────┘
```

**Key Innovation: Everything is Text-to-Text**

```
Classification:
Input: "sentiment: This movie was great"
Output: "positive"

Translation:
Input: "translate English to French: Hello"
Output: "Bonjour"

Summarization:
Input: "summarize: [long article]"
Output: "[summary]"

Question Answering:
Input: "question: Who is CEO? context: John is CEO"
Output: "John"

Unified framework: All tasks have same format (text → text)
```

**Pre-training: Span Corruption**

```
Original: "The cat sat on the mat in the house"
Corrupted: "The <X> on the <Y> in the house"
Targets: "<X> cat sat <Y> mat"

Corrupts random spans (not individual tokens like BERT)
More challenging than MLM
```

**Model Sizes:**

```
T5-Small: 60M params
T5-Base: 220M params
T5-Large: 770M params
T5-3B: 3B params
T5-11B: 11B params

Larger models better, but diminishing returns
```

**Use Cases:**
```
✓ Any text-to-text task (translation, summarization)
✓ Multi-task learning (single model for many tasks)
✓ Consistent API (always text input → text output)

✗ Very large models expensive
✗ Slower than encoder-only or decoder-only for specific tasks
```

### Comparison Summary

| Aspect | BERT | GPT | T5 |
|--------|------|-----|-----|
| **Architecture** | Encoder-only | Decoder-only | Encoder-Decoder |
| **Attention** | Bidirectional | Causal (left-to-right) | Encoder: bidirectional<br>Decoder: causal |
| **Pre-training** | MLM + NSP | Language Modeling | Span Corruption |
| **Best For** | Understanding<br>(classification, NER) | Generation<br>(text, code, dialogue) | Text-to-Text<br>(translation, summarization) |
| **Context** | Full sentence | Left context only | Full context (encoder) |
| **Fine-tuning** | Required (originally) | Optional (GPT-3+) | Required |
| **Size Range** | 110M-340M | 117M-175B+ | 60M-11B |

---

## 7. Fine-Tuning vs Prompt Engineering

### Fine-Tuning

**Definition:** Update model parameters on task-specific data

**Process:**
```
1. Start with pre-trained model (BERT, GPT)
2. Add task-specific head (classification layer)
3. Train on labeled data (1K-100K examples)
4. Update all or some parameters
5. Save fine-tuned model
```

**Example (Sentiment Analysis):**
```
Pre-trained BERT (110M params)
  ↓
Add classification head (768 → 3 classes)
  ↓
Train on 10K labeled reviews
  ↓
Fine-tuned sentiment model

Now: Predict sentiment for new reviews
```

**Pros:**
```
- Best performance (model adapts to specific task)
- Works with small models (BERT-Base sufficient)
- Lower inference cost (smaller fine-tuned model)
- Predictable behavior (trained on your data)
```

**Cons:**
```
- Needs labeled data (1K-10K examples)
- Separate model per task (doesn't scale)
- Training infrastructure required (GPUs)
- Model maintenance (retraining for new data)
```

### Prompt Engineering

**Definition:** Design input prompts to guide pre-trained model (no parameter updates)

**Zero-Shot:**
```
Prompt: "Classify sentiment: This movie was terrible!"
Output: "Negative"

No examples, just instruction
Works with large models (GPT-3, GPT-4)
```

**Few-Shot (In-Context Learning):**
```
Prompt:
"Classify sentiment:
This movie was great! → Positive
This movie was awful! → Negative
This movie was okay. → Neutral
This movie was terrible! →"

Output: "Negative"

Provide examples in prompt
Model learns pattern without training
```

**Pros:**
```
- No labeled data needed (or very little)
- Single model for many tasks
- No training required (instant deployment)
- Easy to update (just change prompt)
```

**Cons:**
```
- Needs large model (GPT-3 size: 175B params)
- Expensive inference (large model calls)
- Less consistent than fine-tuning
- May hallucinate or ignore instructions
```

### When to Use Which?

**Use Fine-Tuning When:**
```
- Have labeled data (1K+ examples)
- Need best performance
- Have training infrastructure
- Task well-defined (sentiment, NER)
- Budget for training but not large inference
- Need predictable behavior
```

**Use Prompt Engineering When:**
```
- Little/no labeled data
- Need flexibility (many tasks, one model)
- No training infrastructure
- Task evolving (instructions change frequently)
- Budget for inference (API calls)
- Prototyping/experimentation
```

**Hybrid Approach:**
```
Parameter-Efficient Fine-Tuning (PEFT):
- LoRA (Low-Rank Adaptation)
- Adapter layers
- Prompt tuning

Idea: Update small subset of parameters
Benefits: Best of both worlds
- Fine-tuning performance
- Prompt engineering flexibility
```

---

## 8. RAG (Retrieval Augmented Generation)

### The Problem LLMs Don't Solve

```
LLM Limitations:
1. Knowledge cutoff (no awareness of recent events)
2. Hallucinations (generates plausible but false info)
3. No access to private data (company documents)
4. No citation/sources (can't verify claims)
```

**Example:**
```
User: "What was our Q3 revenue?"
GPT-4: "I don't have access to your company's financial data"

User: "What happened in the news today?"
GPT-4: "My training data only goes up to April 2023"
```

### RAG Solution

**Idea:** Retrieve relevant documents, provide as context to LLM

```
┌────────────────────────────────────────────────────┐
│                RAG PIPELINE                        │
│                                                     │
│  User Query: "What was Q3 revenue?"                │
│       ↓                                             │
│  ┌──────────────────────┐                         │
│  │  1. RETRIEVAL        │                         │
│  │  Search knowledge    │                         │
│  │  base for relevant   │                         │
│  │  documents           │                         │
│  └──────────────────────┘                         │
│       ↓                                             │
│  Top-K documents:                                  │
│  - Q3 Financial Report                             │
│  - Investor Presentation                           │
│  - Board Meeting Notes                             │
│       ↓                                             │
│  ┌──────────────────────┐                         │
│  │  2. AUGMENTATION     │                         │
│  │  Combine query +     │                         │
│  │  retrieved docs      │                         │
│  └──────────────────────┘                         │
│       ↓                                             │
│  Prompt:                                           │
│  "Given these documents: [docs]                    │
│   Answer: What was Q3 revenue?"                    │
│       ↓                                             │
│  ┌──────────────────────┐                         │
│  │  3. GENERATION       │                         │
│  │  LLM generates       │                         │
│  │  answer from context │                         │
│  └──────────────────────┘                         │
│       ↓                                             │
│  Answer: "Q3 revenue was $12.5M,                   │
│           up 23% YoY. Source: Q3 Report"           │
└────────────────────────────────────────────────────┘
```

### RAG Architecture Components

**Component 1: Knowledge Base**

```
Documents: Company docs, wikis, FAQs, articles
Storage: Vector database (Pinecone, Weaviate, Chroma)

Process:
1. Chunk documents (500-1000 tokens per chunk)
2. Embed chunks (using embedding model)
3. Store embeddings in vector DB

Why chunks?
- LLMs have context limits (4K-32K tokens)
- Retrieval more precise (specific paragraphs)
```

**Component 2: Embedding Model**

```
Purpose: Convert text to dense vectors
Model: OpenAI text-embedding-ada-002, sentence-transformers

Example:
"Q3 revenue increased 23%" → [0.2, -0.4, 0.7, ..., 0.1]  (1536-dim)

Similar documents → Similar embeddings
Enables semantic search (not just keyword matching)
```

**Component 3: Retrieval (Vector Search)**

```
Query: "What was Q3 revenue?"
  ↓
Embed query: [0.3, -0.2, 0.6, ..., 0.2]
  ↓
Cosine similarity with all doc embeddings
  ↓
Top-K most similar chunks (K=3-10):
  1. "Q3 revenue: $12.5M, up 23% YoY..." (similarity: 0.92)
  2. "Quarterly performance exceeded..." (similarity: 0.85)
  3. "Revenue breakdown by segment..." (similarity: 0.78)
```

**Component 4: Prompt Construction**

```
System prompt:
"You are a helpful assistant. Answer based on provided context.
If answer not in context, say 'I don't know'."

User prompt:
"Context:
[Document 1]: Q3 revenue was $12.5M...
[Document 2]: Quarterly performance...
[Document 3]: Revenue breakdown...

Question: What was Q3 revenue?
Answer:"

LLM generates answer based on context
```

**Component 5: LLM Generation**

```
Input: Augmented prompt (query + retrieved docs)
Model: GPT-4, Claude, or open-source (Llama)
Output: Answer with citations

"Q3 revenue was $12.5M, up 23% YoY. [Source: Q3 Financial Report]"
```

### RAG Benefits

```
1. Up-to-date information:
   - Knowledge base updated anytime
   - No model retraining needed

2. Private data:
   - Works with company-specific documents
   - LLM doesn't need to be trained on private data

3. Citations:
   - Can trace answer to source documents
   - Verifiable, trustworthy

4. Reduced hallucinations:
   - Answer must be in retrieved docs
   - "I don't know" if not in context

5. Cost-effective:
   - Don't need to fine-tune LLM
   - Update knowledge by updating docs (cheap)
```

### RAG Challenges

**1. Retrieval Quality:**
```
Problem: If wrong documents retrieved, answer will be wrong
"Garbage in, garbage out"

Solutions:
- Better embedding models (fine-tuned on domain)
- Hybrid search (vector + keyword)
- Re-ranking (second-stage filtering)
```

**2. Context Length:**
```
Problem: LLMs have token limits (4K-32K)
Can't provide 100 pages of documents

Solutions:
- Chunking (only provide relevant chunks)
- Summarization (summarize long docs first)
- Hierarchical retrieval (outline → details)
```

**3. Latency:**
```
Retrieval + Generation = 2-5 seconds
Slower than direct LLM call

Solutions:
- Cache frequent queries
- Async retrieval (stream results)
- Smaller LLMs (faster inference)
```

**4. Inconsistency:**
```
Problem: LLM may ignore context or hallucinate anyway

Solutions:
- Prompt engineering ("Only use context provided")
- Constrained decoding (force answer from context)
- Post-processing (verify answer in retrieved docs)
```

---

## 9. LLM Infrastructure Requirements

### Compute Requirements

**Training:**

```
GPT-3 (175B parameters):
- Hardware: 10,000 V100 GPUs
- Duration: 1 month
- Cost: $4-12 million
- Energy: 1,287 MWh (equivalent to 120 homes for a year)

Conclusion: Training from scratch is for large organizations only
Most companies: Use pre-trained models (OpenAI API, open-source)
```

**Inference:**

```
Single GPT-3 call:
- Latency: 1-2 seconds
- Compute: 1-2 A100 GPUs (175B params doesn't fit on single GPU)
- Cost: $0.002 per 1K tokens (OpenAI pricing)

At scale (1M requests/day):
- GPUs needed: 10-50 A100s
- Cost: $2,000/day (OpenAI) or $500K-2M/year (self-hosted GPUs)
```

### Memory Requirements

**Model Size vs GPU Memory:**

```
Model parameters stored in:
- FP32: 4 bytes per parameter
- FP16: 2 bytes per parameter
- INT8: 1 byte per parameter (quantized)

GPT-3 (175B params):
- FP32: 175B × 4 bytes = 700 GB
- FP16: 175B × 2 bytes = 350 GB
- INT8: 175B × 1 byte = 175 GB

GPU Memory:
- A100: 80 GB
- V100: 32 GB

To load GPT-3 (FP16):
350 GB / 80 GB = 5 A100 GPUs minimum
(In practice, need more for activations, KV cache)
```

**KV Cache (Key-Value Cache):**

```
Problem:
Generating 100-token response = 100 forward passes
Each pass recomputes attention for all previous tokens
Wasteful!

Solution: KV Cache
Store computed keys and values
Reuse for subsequent tokens

Memory cost:
Per token: 2 × layers × hidden_dim × 2 bytes (K and V, FP16)
100 tokens: ~10 MB per sample

Batch of 32: 320 MB just for KV cache
```

### Optimization Techniques

**1. Model Quantization:**

```
FP32 → INT8: 4× smaller model
Performance: ~1-2% accuracy loss
Benefits: Fits on fewer GPUs, faster inference

Tools: bitsandbytes, GPTQ, AWQ
```

**2. Model Distillation:**

```
Train small model to mimic large model
Example: DistilBERT (66M) mimics BERT (110M)
Result: 40% smaller, 60% faster, 97% performance

Use case: Deploy small model for inference
Train/distill once, fast inference forever
```

**3. Flash Attention:**

```
Optimized attention computation
Reduces memory usage 2-4×
Speeds up inference 2-3×

How: Fused kernels, tiling, recomputation
Standard in modern frameworks (PyTorch 2.0+)
```

**4. Tensor Parallelism:**

```
Split model across multiple GPUs
Each GPU has fraction of model

Example: 175B model across 8 GPUs
Each GPU: ~22B parameters

Requires: Fast GPU interconnect (NVLink, Infiniband)
```

**5. Batch Processing:**

```
Process multiple requests together
Amortize model loading cost

Example:
Single request: 50 ms per token
Batch of 32: 60 ms per token (for all 32!)

Throughput: 32× higher
Latency: Slightly higher (wait for batch to fill)
```

---

## 10. LLM Deployment Patterns

### Pattern 1: API-Based (OpenAI, Anthropic)

```
Application → OpenAI API → GPT-4 → Response

Pros:
- No infrastructure management
- Always latest model
- Pay-per-use (no upfront cost)
- Scales automatically

Cons:
- Expensive at scale ($0.03 per 1K tokens)
- Data privacy concerns (sent to third party)
- Latency (network call)
- Vendor lock-in
```

**When to use:**
```
- Prototype/MVP
- Low volume (<1M requests/month)
- Don't have ML infrastructure
- Need latest models (GPT-4, Claude)
```

### Pattern 2: Self-Hosted Open-Source

```
Application → Load Balancer → LLM Servers (Llama 2, Falcon) → Response

Pros:
- Full control (data stays in-house)
- Lower cost at scale (hardware amortized)
- Customization (fine-tuning)
- No vendor dependency

Cons:
- Infrastructure management (GPUs, deployment)
- Upfront cost (GPUs expensive)
- Maintenance (model updates, optimization)
- May lag behind latest proprietary models
```

**When to use:**
```
- High volume (>10M requests/month)
- Data privacy critical (finance, healthcare)
- Need customization (domain-specific)
- Have ML infrastructure team
```

### Pattern 3: Hybrid (RAG + API)

```
User Query
  ↓
Retrieve from local knowledge base (Pinecone, Weaviate)
  ↓
Augment with retrieved docs
  ↓
Send to OpenAI API (GPT-4)
  ↓
Response with citations

Pros:
- Best of both worlds
- Private data + powerful LLM
- No training needed

Cons:
- Two systems to manage (vector DB + LLM API)
- Latency (retrieval + generation)
```

**When to use:**
```
- Need to query private documents
- Want latest LLM capabilities
- Don't want to fine-tune
```

### Pattern 4: Fine-Tuned Model

```
Application → Fine-tuned GPT-3.5/Llama → Response

Pros:
- Best task performance
- Smaller model (faster, cheaper)
- Consistent behavior

Cons:
- Needs training data
- Fine-tuning cost/time
- Separate model per task
```

**When to use:**
```
- Have labeled data (1K-10K examples)
- Single well-defined task
- Need optimal performance
```

---

## 11. Interview Focus: LLM Deployment Architecture

### Q1: "Design a customer support chatbot using LLMs. 100K queries/day, access to internal knowledge base."

**Good Answer:**

"I'd design a hybrid RAG architecture with caching and fallback mechanisms:

**High-Level Architecture:**

```
User Query
  ↓
┌─────────────────────────────┐
│  1. CACHE LAYER (Redis)     │
│  Check if query answered    │
│  before                      │
└─────────────────────────────┘
  ↓ (cache miss)
┌─────────────────────────────┐
│  2. INTENT CLASSIFIER       │
│  Route to appropriate path  │
│  (FAQ vs complex query)     │
└─────────────────────────────┘
  ↓ (complex query)
┌─────────────────────────────┐
│  3. RAG RETRIEVAL           │
│  Vector DB (Pinecone)       │
│  Retrieve top-5 docs        │
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│  4. LLM GENERATION          │
│  GPT-4 (complex)            │
│  GPT-3.5 (simple)           │
│  Generate answer + citation │
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│  5. SAFETY LAYER            │
│  Content filtering          │
│  PII detection              │
└─────────────────────────────┘
  ↓
Response + Cache
```

**Component Details:**

**1. Cache Layer (Redis):**
```
Purpose: Reduce LLM calls for repeated queries
Implementation:
- Key: Query embedding (similar queries hit same cache)
- Value: Generated response + metadata
- TTL: 1 hour (fresh answers)

Impact:
- Cache hit rate: 30-40% (common support questions)
- Latency: <10ms for cache hits
- Cost: Save $300/day (30K fewer LLM calls)
```

**2. Intent Classification (Small BERT Model):**
```
Purpose: Route queries efficiently
Classes:
- FAQ (simple, use template)
- Knowledge_base (RAG)
- Escalate_human (complex/sensitive)

Model: DistilBERT fine-tuned on support queries
Latency: 20ms
Accuracy: 92%

Routing:
- FAQ: Return template (no LLM call)
- Knowledge_base: RAG pipeline
- Escalate: Human handoff
```

**3. RAG Retrieval (Pinecone):**
```
Knowledge base:
- Product docs, FAQs, support articles (10K documents)
- Chunked (500 tokens), embedded (OpenAI ada-002)
- Updated nightly (new docs, updated info)

Retrieval:
- Hybrid search (vector + keyword)
- Top-5 chunks (most relevant)
- Re-ranking (cross-encoder for precision)
- Latency: 100-150ms
```

**4. LLM Generation (Tiered):**
```
Simple queries: GPT-3.5-turbo
- Price: $0.0015 per 1K tokens
- Latency: 500ms
- 80% of queries

Complex queries: GPT-4
- Price: $0.03 per 1K tokens
- Latency: 1-2s
- 20% of queries

Prompt template:
"You are a customer support agent.
Answer based on provided context.
If answer not in context, say 'I'll escalate to human agent'.
Be concise and friendly.

Context: [retrieved docs]
Question: [user query]
Answer:"

Generation config:
- Temperature: 0.3 (consistent, less creative)
- Max tokens: 150 (concise answers)
- Stop sequences: ["\n\nHuman:", "Question:"]
```

**5. Safety Layer:**
```
Content filtering:
- Toxic language detection (perspective API)
- PII detection (remove SSNs, credit cards)
- Prompt injection defense (detect malicious prompts)

Guardrails:
- Answer must cite source (no hallucination)
- If confidence <0.7, escalate to human
- Flag sensitive topics (refunds, account deletion)
```

**Infrastructure:**

```
Load Balancer (AWS ALB)
  ↓
Application Servers (ECS, 10 containers)
  ├─ Redis cluster (3 nodes, 16GB)
  ├─ Pinecone (standard tier)
  ├─ OpenAI API (rate limit: 1000 RPM)
  └─ Monitoring (CloudWatch, Datadog)

Scaling:
- Auto-scale on CPU >70%
- Rate limiting: 100 requests/min per user
- Queue for bursts (SQS, process async)
```

**Cost Analysis (100K queries/day):**

```
Breakdown:
- OpenAI API (70K after cache): $2,100/day
  • GPT-3.5 (56K): $252/day
  • GPT-4 (14K): $1,848/day
- Pinecone: $70/day
- Redis: $10/day
- Compute (ECS): $50/day

Total: ~$2,230/day = $67K/month

Optimizations to reduce cost:
- Increase cache hit rate (40% → 50%): Save $300/day
- Use smaller model for simple queries: Save $500/day
- Self-host open-source LLM (Llama 2): Save $1,500/day (but $20K setup)
```

**Monitoring & Metrics:**

```
Real-time:
- Latency (P50, P95, P99)
- Error rate
- LLM API status
- Cache hit rate

Daily:
- User satisfaction (thumbs up/down)
- Escalation rate (to human)
- Cost per query
- Answer accuracy (human review sample)

Weekly:
- Knowledge base coverage (% queries with relevant docs)
- Model drift (accuracy over time)
- Popular queries (improve FAQs)
```

**Fallback Mechanisms:**

```
If OpenAI API down:
1. Serve cached responses (if available)
2. Fall back to rule-based system (keyword matching)
3. Escalate all to human queue
4. Show graceful error (not "500 Internal Server Error")

If vector DB down:
1. Skip retrieval, use LLM directly (less accurate)
2. Or escalate to human

Goal: Degrade gracefully, never hard failure
```

**Expected Performance:**

```
Metrics:
- Latency: P95 < 2s (including retrieval + generation)
- Accuracy: 85% (human review)
- Automated resolution: 70% (no human needed)
- User satisfaction: 4.2/5

Business impact:
- Deflect 70K queries/day from human agents
- Average handling time: 5 min/query
- Savings: 70K × 5 min = 5,833 hours/day
- At $20/hour: $116K/day = $3.5M/month savings

ROI: $3.5M savings - $67K cost = $3.43M net/month
```

This architecture balances cost, performance, and accuracy while maintaining graceful degradation and explainability."

---

### Q2: "How would you optimize LLM inference latency from 3 seconds to <500ms?"

**Good Answer:**

"I'd attack latency from multiple angles:

**Current Baseline (3 seconds):**
```
Breakdown:
- Model loading: 500ms
- Input processing: 100ms
- Generation (100 tokens): 2,000ms (20ms/token)
- Post-processing: 100ms
- Network overhead: 300ms
```

**Optimization Strategy:**

**1. Model Optimization (Target: 1s savings):**

**Quantization:**
```
Current: FP16 (2 bytes/param)
Optimized: INT8 (1 byte/param)

Benefits:
- 2× smaller model → fits in fewer GPUs
- 2× faster matrix multiplication
- Negligible accuracy loss (<1%)

Tool: bitsandbytes, GPTQ
Result: 2000ms → 1200ms (save 800ms)
```

**Flash Attention:**
```
Current: Standard attention (memory-bound)
Optimized: Flash Attention (fused kernels)

Benefits:
- 2-3× faster attention computation
- Lower memory usage (no intermediate tensors)

Tool: xformers, PyTorch 2.0+
Result: 1200ms → 800ms (save 400ms)
```

**2. Smaller Model (Target: 500ms savings):**

**Model Selection:**
```
Current: GPT-3 (175B params)
Optimized: Llama 2 7B or Mistral 7B

Why:
- 25× smaller (175B → 7B)
- Single GPU instead of 8
- 5-10× faster inference

Trade-off: ~5-10% accuracy loss
Mitigation: Fine-tune on domain data (recover accuracy)

Result: 800ms → 300ms (save 500ms)
```

**3. Caching (Target: Eliminate 40% of calls):**

**Semantic Caching:**
```
Cache not just exact queries, but similar queries

Example:
Cache: "What's your refund policy?" → [response]
Query: "How do I get a refund?" → Cache hit! (similar embedding)

Implementation:
- Embed query (5ms)
- Cosine similarity with cache (10ms)
- If similarity >0.95, return cached response

Impact:
- 40% cache hit rate
- Effective latency: 0.6 × 300ms = 180ms average
```

**4. Batch Processing (Target: Higher throughput):**

**Dynamic Batching:**
```
Instead of processing one request at a time, batch multiple

Current: 1 request → 300ms
Batched: 10 requests → 350ms total (35ms per request effective)

How:
- Wait 50ms to collect requests
- Process batch together
- Amortize model overhead

Trade-off: 50ms added latency (waiting) but 10× throughput
```

**5. Speculative Decoding (Target: 100ms savings):**

```
Idea: Use small fast model to draft tokens, large model to verify

Process:
1. Small model (1B params) generates 5 tokens (20ms)
2. Large model (7B params) verifies all at once (50ms)
3. If verified, save 5 × 20ms = 100ms vs serial

Result: 300ms → 200ms (save 100ms)

Tool: Speculative decoding (built into vLLM, TensorRT-LLM)
```

**6. Early Stopping (Target: Variable savings):**

```
Stop generation when quality sufficient, not at max tokens

Example:
Question: "What's your refund policy?"
Good answer: 30 tokens (sufficient)
Max tokens: 100 (wasteful)

Savings: 70 tokens × 2ms/token = 140ms

Implementation:
- Confidence threshold (perplexity <2.5 → stop)
- Semantic completeness (answer addresses question)
```

**7. KV Cache Optimization (Target: 50ms savings):**

```
Cache keys and values during generation (standard practice)
But optimize memory layout:

PagedAttention (vLLM):
- Efficient memory management
- 2-4× higher throughput
- Same latency but handle more concurrent requests

Result: Don't block waiting for memory
```

**8. Network Optimization (Target: 100ms savings):**

**Co-location:**
```
Place LLM server in same region/AZ as application
Reduce network latency: 300ms → 200ms (save 100ms)
```

**HTTP/2:**
```
Use HTTP/2 or gRPC instead of HTTP/1.1
Multiplexing, header compression
Save 50ms in overhead
```

**9. Infrastructure (Target: Consistent low latency):**

**GPU Selection:**
```
Current: V100 (older)
Optimized: A100 or H100 (newer, 2-3× faster)

Also consider:
- Tensor RT (NVIDIA inference engine)
- AWS Inferentia (custom ML chip, cheaper)
```

**Continuous Batching (vLLM):**
```
New requests added to batch continuously
Don't wait for entire batch to finish

Result: Lower latency variance (more predictable)
```

**Final Optimized Pipeline:**

```
Latency Breakdown:
- Semantic cache check: 15ms
  • Cache hit (40%): Return (15ms total) ✓
  • Cache miss (60%): Continue
- Model inference (Llama 2 7B, INT8, Flash Attention): 180ms
- Post-processing: 50ms
- Network (co-located): 100ms

Total (cache miss): 345ms ✓ (under 500ms target!)
Average (with cache): 0.4 × 15ms + 0.6 × 345ms = 213ms ✓✓
```

**Cost vs Performance Trade-offs:**

```
Approach | Latency | Cost | Accuracy
---------|---------|------|----------
Baseline (GPT-3) | 3000ms | $0.03/1K | 95%
Quantization | 1200ms | $0.015/1K | 94%
Smaller model (7B) | 300ms | $0.002/1K | 88%
+ Fine-tuning | 300ms | $0.002/1K | 92%
+ Caching | 200ms avg | $0.001/1K | 92%

Recommendation: Smaller model + fine-tuning + caching
- 10× faster
- 30× cheaper
- 3% accuracy loss (acceptable for most use cases)
```

**Monitoring for Latency:**

```
Metrics:
- P50, P95, P99 latency (track tail latencies)
- Cache hit rate (aim for >40%)
- Tokens per second (throughput)
- GPU utilization (should be >80%)

Alerts:
- P95 latency >500ms
- Cache hit rate <30%
- GPU utilization <50% (underutilized)
```

This multi-pronged approach achieves <500ms latency while maintaining good accuracy and reducing costs 30×."

---

## Summary: Key Takeaways

**1. Transformer Architecture:**
- Self-attention: Every word attends to every other word (O(n²) complexity)
- Multi-head attention: Multiple representation subspaces (8-16 heads)
- Positional encoding: Injects position information (sine/cosine or learned)
- Residual connections: Enable deep networks (like ResNet)

**2. Model Families:**
- **BERT:** Encoder-only, bidirectional, understanding tasks (classification, NER, QA)
- **GPT:** Decoder-only, causal, generation tasks (text completion, dialogue, code)
- **T5:** Encoder-decoder, text-to-text, versatile (translation, summarization)

**3. Fine-Tuning vs Prompting:**
- **Fine-tuning:** Best performance, needs data, separate model per task
- **Prompt engineering:** Flexible, no training, needs large model (expensive inference)
- **Hybrid (PEFT):** LoRA, adapters - update small subset of parameters

**4. RAG:**
- Retrieval: Vector search in knowledge base (Pinecone, Weaviate)
- Augmentation: Combine query + retrieved docs in prompt
- Generation: LLM generates answer with citations
- Benefits: Up-to-date, private data, reduced hallucinations

**5. LLM Infrastructure:**
- **Training:** Extremely expensive (millions of dollars, 10K GPUs)
- **Inference:** Challenging (large memory, high latency)
- **Optimization:** Quantization, distillation, Flash Attention, batching, caching

**6. Deployment Patterns:**
- **API-based:** Easy, expensive at scale, privacy concerns (OpenAI, Anthropic)
- **Self-hosted:** Control, cheaper at scale, infrastructure overhead (Llama, Falcon)
- **Hybrid (RAG):** Private data + powerful LLM (best of both)

**For Interviews:**
- Understand attention mechanism deeply (can explain with examples)
- Know trade-offs: BERT vs GPT, fine-tuning vs prompting
- RAG architecture (retrieval, augmentation, generation components)
- Latency optimization strategies (quantization, caching, smaller models)
- Cost analysis (API vs self-hosted, throughput vs latency)
- Systems thinking (caching, load balancing, fallbacks, monitoring)

**Critical Interview Topics:**
- LLM deployment architecture (RAG + caching + tiered models)
- Inference optimization (quantization, batching, Flash Attention)
- Cost-performance trade-offs (accuracy vs latency vs cost)
- Production considerations (monitoring, safety, graceful degradation)

---

**END OF DAY 6**

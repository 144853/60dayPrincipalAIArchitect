# Day 5: NLP & Text Processing - Interview Questions

## Section 1: NLP Fundamentals & Preprocessing

### Basic Concepts (1-3 years experience)

**Q1: Why is text processing fundamentally different from image processing in deep learning?**

**Expected Answer:**
- **Discrete vs. Continuous:** Images are continuous pixel values (0-255), while text consists of discrete tokens (words/characters). You can't slightly change a word like you can a pixel; "cat" to "bat" is a complete semantic jump.
- **Sequential Nature:** In text, order strictly determines meaning ("Dog bites man" ≠ "Man bites dog"). While spatial relationships matter in images, text is inherently sequential and temporal.
- **Variable Length:** Sentences vary wildly in length, whereas neural networks typically expect fixed-size inputs (requiring padding/packing).
- **Vocabulary Size:** Text involves mapping high-dimensional sparse vectors (vocabularies of 50k+) to dense vectors, unlike fixed channel inputs in images.

**Q2: Explain the difference between Stemming and Lemmatization. When would you use one over the other?**

**Expected Answer:**
- **Stemming:** A crude heuristic process that chops off the ends of words (e.g., "running" -> "run", "better" -> "better"). It's fast but often produces non-words.
- **Lemmatization:** Uses vocabulary and morphological analysis to return the dictionary form of a word (e.g., "better" -> "good"). It requires POS tagging and is computationally heavier.
- **Usage:** Use stemming for simple search/retrieval where speed is critical and exact meaning is less important. Use lemmatization for tasks requiring semantic understanding like Question Answering or Chatbots.
- **Modern Context:** In modern Transformer models, we often skip both and use subword tokenization.

**Q3: What is Subword Tokenization (like BPE or WordPiece) and why is it the standard for modern NLP?**

**Expected Answer:**
- **Problem it solves:** Traditional word-level tokenization suffers from large vocabulary sizes and the "Out-Of-Vocabulary" (OOV) problem for unseen words. Character-level is too granular and produces long sequences.
- **How it works:** It breaks words into frequent sub-units. Common words remain whole ("apple"), while rare words are split ("unhappiness" -> "un" + "happi" + "ness").
- **Benefits:**
  1. **Fixed Vocabulary:** Keeps vocabulary size manageable (e.g., 30k-50k).
  2. **No OOV:** Can represent any word by constructing it from subwords.
  3. **Morphology:** Captures meaning of parts (e.g., "ing", "ed", "pre").

### Intermediate Questions (3-5 years experience)

**Q4: Compare TF-IDF with Word Embeddings (like Word2Vec). What are the limitations of TF-IDF?**

**Expected Answer:**
- **TF-IDF (Sparse):** Represents documents based on word frequency counts weighted by rarity.
  - *Pros:* Fast, interpretable, good for keyword search.
  - *Cons:* High dimensionality, sparse (mostly zeros), captures **no semantic meaning** (e.g., doesn't know "car" and "automobile" are similar).
- **Word Embeddings (Dense):** Represents words as dense vectors where similar words are close in vector space.
  - *Pros:* Low dimensionality, captures semantic relationships and analogies.
  - *Cons:* Requires training, less interpretable dimensions.

**Q5: How does FastText improve upon Word2Vec?**

**Expected Answer:**
- **Word2Vec limitation:** It learns a distinct vector for each word. If a word wasn't in the training set (OOV), it has no vector. It ignores internal word structure.
- **FastText innovation:** Represents each word as a bag of character n-grams. The vector for a word is the sum of its n-gram vectors.
- **Advantage:** It can generate embeddings for **unseen words** (OOV) by summing the vectors of their n-grams. It works exceptionally well for morphologically rich languages or noisy text (typos).

---

## Section 2: RNNs, Attention, and Transformers

### Basic Concepts

**Q6: What is the "Vanishing Gradient" problem in RNNs, and how do Transformers avoid it?**

**Expected Answer:**
- **RNN Problem:** In RNNs, information must pass sequentially through every time step. For long sequences, gradients (signals) from the end of the sentence become tiny by the time they reach the beginning during backpropagation, causing the model to "forget" early context.
- **Transformer Solution:** Transformers process the entire sequence in parallel. The **Attention mechanism** allows any position to look directly at any other position. The path length between any two words is always 1, regardless of distance, eliminating the vanishing gradient problem for long-range dependencies.

**Q7: Explain the concept of "Self-Attention" in simple terms.**

**Expected Answer:**
- Self-attention allows a model to look at other words in the input sentence to understand the current word better.
- **Example:** In "The animal didn't cross the street because **it** was too tired", self-attention allows the model to associate "**it**" strongly with "**animal**" rather than "street".
- **Mechanism:** It computes a weighted sum of all words in the sentence, where the weights determine how relevant each word is to the current word being processed.

### Advanced Questions

**Q8: Why do Transformers need Positional Encodings?**

**Expected Answer:**
- Unlike RNNs, which process data sequentially (implicitly knowing word order), Transformers process all words in parallel.
- Without positional encodings, a Transformer would see "Dog bites man" and "Man bites dog" as identical "bags of words" because the self-attention mechanism is permutation-invariant.
- Positional encodings inject information about the *position* of tokens into the embeddings so the model can learn order and relative distances.

**Q9: Explain the difference between the Encoder and Decoder blocks in a Transformer.**

**Expected Answer:**
- **Encoder (e.g., BERT):**
  - Uses **bidirectional** self-attention (can see future and past tokens).
  - Goal: Create a rich representation of the input.
  - Use case: Understanding tasks (Classification, NER, QA).
- **Decoder (e.g., GPT):**
  - Uses **masked** self-attention (can only see past tokens, not future).
  - Goal: Generate the next token.
  - Use case: Generation tasks (Text completion, Translation).

---

## Section 3: BERT vs. GPT & Modern LLMs

### Scenario-Based Questions

**Q10: You need to build a system to extract company names and dates from legal contracts. Would you use BERT or GPT? Why?**

**Expected Answer:**
- **Choice:** **BERT** (or a variant like RoBERTa/DistilBERT).
- **Reasoning:**
  1. **Task Type:** This is Named Entity Recognition (NER), which is a token classification/understanding task.
  2. **Context:** BERT is bidirectional; to understand a token, it looks at both the words before and after it, which is crucial for extraction accuracy.
  3. **Efficiency:** For extraction, a fine-tuned BERT model is typically faster and cheaper to run than prompting a massive GPT model.
  4. **Output:** We need precise span extraction, not open-ended text generation.

**Q11: You are building a customer support chatbot that needs to generate natural-sounding responses to user queries. Which architecture fits best?**

**Expected Answer:**
- **Choice:** **GPT** (Decoder-only architecture).
- **Reasoning:**
  1. **Task Type:** This is a text generation task.
  2. **Mechanism:** GPT is autoregressive, designed specifically to predict the next word in a sequence, making it ideal for fluent, coherent text generation.
  3. **Capability:** Modern LLMs (GPT-3.5/4) have strong few-shot capabilities to follow instructions and maintain conversation history.

**Q12: Explain the pre-training objectives of BERT (MLM) vs. GPT (Causal LM).**

**Expected Answer:**
- **BERT (Masked Language Modeling - MLM):**
  - *Task:* Randomly mask 15% of tokens in the input and ask the model to predict them based on the surrounding context (left and right).
  - *Goal:* Learn deep bidirectional context and understanding.
- **GPT (Causal Language Modeling - CLM):**
  - *Task:* Predict the next word in the sequence given ONLY the previous words.
  - *Goal:* Learn to generate coherent text sequences.

---

## Section 4: NLP System Design

### Q13: Design a Sentiment Analysis system for a high-volume Twitter feed (10k tweets/sec).

**Expected Answer:**

**1. Requirements:**
- **Input:** Real-time tweets.
- **Output:** Sentiment score (Positive/Negative/Neutral).
- **Constraint:** High throughput (10k/sec), low latency.

**2. Data Pipeline:**
- **Ingestion:** Kafka to handle the high throughput stream.
- **Preprocessing:** Minimal cleaning (remove HTML, maybe handle @mentions). Use Subword tokenization (handles hashtags/slang well).

**3. Model Selection:**
- **Option A (Accuracy):** Fine-tuned **DistilBERT**. It's 40% smaller and 60% faster than BERT, retaining 95% performance.
- **Option B (Speed):** If DistilBERT is too slow/expensive for 10k/sec, use **FastText** or a simple **Bi-LSTM**.
- **Decision:** Start with DistilBERT. If latency is an issue, use **Model Distillation** to train a smaller student model (like a tiny LSTM) to mimic the BERT teacher, or use **Quantization** (FP16/INT8).

**4. Deployment:**
- Deploy model on Kubernetes with auto-scaling.
- Use **Batching** (process tweets in groups of 32/64) to maximize GPU utilization.
- Use **ONNX Runtime** or **TensorRT** for optimized inference.

**5. Handling Drift:**
- Twitter language changes fast (new slang). Monitor confidence scores. Retrain weekly on active learning samples (tweets where model had low confidence).

---

### Q14: How would you build a "Semantic Search" engine for a company's internal documentation?

**Expected Answer:**

**1. The Problem:** Keyword search (TF-IDF/Elasticsearch) fails when users use synonyms (e.g., searching "vacation policy" but docs say "time off").

**2. Solution: Dense Retrieval (Embeddings)**

**Indexing Phase:**
- Break documents into chunks (passages).
- Use a **Bi-Encoder** (like SBERT - Sentence-BERT) to convert each chunk into a fixed-size vector.
- Store vectors in a **Vector Database** (Pinecone, Milvus, or FAISS).

**Search Phase:**
- Convert user query into a vector using the same SBERT model.
- Perform **Approximate Nearest Neighbor (ANN)** search in the vector DB to find most similar document vectors (Cosine Similarity).

**3. Re-Ranking (Optional but recommended):**
- Retrieve top 50 candidates using the fast vector search.
- Pass the query + candidate pairs through a **Cross-Encoder** (more accurate but slower BERT model) to re-rank them for final precision.

**4. Handling Domain Specificity:**
- Pre-trained BERT might not know company jargon.
- **Fine-tune** the embedding model using domain data (triplets of: query, positive_doc, negative_doc) if available.

---

## Section 5: Practical Debugging & Production

### Q15: Your NER model is failing to recognize names in a specific region (e.g., Indian names), despite having high overall accuracy. How do you fix this?

**Expected Answer:**

**Diagnosis:**
- The model likely has **bias** in the training data (e.g., trained mostly on Western names).
- Subword tokenization might be splitting unfamiliar names into meaningless chunks that the model hasn't learned to associate with the "PERSON" entity.

**Solutions:**
1. **Data Augmentation:** Collect more labeled data containing Indian names. If manual labeling is expensive, use **weak supervision** (programmatically replace Western names in existing training data with Indian names to create synthetic examples).
2. **Transfer Learning:** Fine-tune a **multilingual model** (like mBERT or XLM-RoBERTa) which has seen more diverse names during pre-training.
3. **Gazetteers:** Inject external knowledge. Add a feature indicating if a token appears in a list of known Indian names (though this is less "pure" deep learning, it's practical).
4. **Active Learning:** Specifically sample low-confidence predictions on Indian documents for human review and retraining.

### Q16: You have a limited budget for labeling data. You need to train a classifier. What strategy do you use?

**Expected Answer:**
I would use **Active Learning** combined with **Transfer Learning**.

1. **Zero-Shot/Few-Shot:** Start with a large pre-trained model (like GPT-4 or a Zero-shot classifier) to auto-label the dataset. This gives a noisy baseline.
2. **Train a smaller model:** Train a DistilBERT model on this noisy data.
3. **Active Learning Loop:**
   - Run the model on unlabeled data.
   - Select samples where the model is **least confident** (entropy is high).
   - Send ONLY these hard samples to humans for labeling.
   - Retrain the model.
4. **Result:** You achieve high accuracy with a fraction of the labeled data because humans only label the edge cases, not the "easy" ones.

---

## Section 6: Coding & Implementation (Conceptual)

### Q17: How do you handle input text that is longer than the BERT limit (512 tokens)?

**Expected Answer:**
There are three common strategies:

1. **Truncation:** Keep only the first 512 tokens (head) or the last 512 (tail).
   - *Pros:* Simple.
   - *Cons:* Loss of information. Good for news (lead paragraph), bad for legal docs.
2. **Chunking (Sliding Window):**
   - Split text into overlapping chunks (e.g., tokens 0-512, 256-768, etc.).
   - Run BERT on each chunk independently.
   - **Aggregation:** For classification, average the probabilities or take the max. For extraction, combine spans.
   - *Pros:* No info loss.
   - *Cons:* Slower (multiple inference passes), context broken at boundaries.
3. **Long-Context Models:** Use architectures designed for long sequences, like **Longformer** or **BigBird**, which use sparse attention mechanisms to handle 4096+ tokens.

### Q18: Implement a simple "Bag of Words" vectorizer from scratch in Python (pseudocode).

**Expected Answer:**
```python
class BagOfWords:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, documents):
        # Build vocabulary
        for doc in documents:
            tokens = doc.lower().split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.vocab_size
                    self.vocab_size += 1

    def transform(self, documents):
        vectors = []
        for doc in documents:
            # Create zero vector
            vec = [0] * self.vocab_size
            tokens = doc.lower().split()
            for token in tokens:
                if token in self.vocab:
                    idx = self.vocab[token]
                    vec[idx] += 1
            vectors.append(vec)
        return vectors
```

---

## Evaluation Rubric

### Junior Level
- Understands tokenization, stemming vs lemmatization.
- Knows the difference between Bag of Words and Embeddings.
- Can explain basic metrics (Precision/Recall).

### Mid Level
- Deep understanding of Word2Vec/FastText.
- Can explain Attention and Transformer basics.
- Knows when to use BERT vs GPT.
- Can implement basic pipelines using libraries like HuggingFace.

### Senior/Principal Level
- Can design end-to-end systems (latency, cost, scale).
- Understands architectural trade-offs (Encoder vs Decoder, Sparse vs Dense).
- Can solve production issues (drift, bias, long documents).
- Knows how to optimize for inference (distillation, quantization).
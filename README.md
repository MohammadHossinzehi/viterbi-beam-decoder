# Viterbi Decoder with Beam Search

A production-grade implementation of the Viterbi algorithm and beam search decoder for optimal sequence labeling in Hidden Markov Models and other probabilistic sequence models.

## What It Does

This library provides two algorithms for finding the most likely sequence of hidden states given a sequence of observations:

1. **Standard Viterbi Algorithm**: Guarantees optimal solution using dynamic programming
   - O(n x |S|^2) time complexity
   - O(n x |S|) space complexity
   - Works in log-space to avoid numerical underflow

2. **Beam Search Variant**: Memory-efficient approximation
   - Keeps only the k best hypotheses at each step
   - Useful for very long sequences or large state spaces
   - Gracefully degrades to Viterbi as beam width increases

## Why Use This?

- **Numerically Stable**: Log-space computation prevents underflow with small probabilities
- **Well-Tested**: Comprehensive unit test suite with edge cases
- **Flexible**: Supports custom emission and transition probabilities
- **Educational**: Clean, documented code ideal for learning
- **Production-Ready**: Type hints and error handling throughout

## Installation

Clone the repository:

```bash
git clone https://github.com/MohammadHossinzehi/viterbi-beam-decoder.git
cd viterbi-beam-decoder
```

## Usage

### Basic Example: Weather Prediction

```python
from viterbi_decoder import ViterbiDecoder

# Define states and observations
states = ["Sunny", "Rainy"]
observations = ["happy", "grumpy", "grumpy", "happy", "happy"]

# Define model parameters
start_probs = {"Sunny": 0.8, "Rainy": 0.2}

# P(next_state | current_state)
trans_probs = {
    ("Sunny", "Sunny"): 0.7,
    ("Sunny", "Rainy"): 0.3,
    ("Rainy", "Sunny"): 0.4,
    ("Rainy", "Rainy"): 0.6
}

# P(observation | state)
emit_probs = {
    ("Sunny", "happy"): 0.8,
    ("Sunny", "grumpy"): 0.2,
    ("Rainy", "happy"): 0.4,
    ("Rainy", "grumpy"): 0.6
}

# Decode using Viterbi
decoder = ViterbiDecoder(states, log_probs=True)
path, score = decoder.viterbi(observations, start_probs, emit_probs, trans_probs)

print(f"Most likely weather sequence: {path}")
print(f"Path probability (log): {score}")
```

### Beam Search for Long Sequences

```python
# For very long sequences, use beam search to reduce memory usage
beam_path, beam_score = decoder.beam_search(
    observations,
    start_probs,
    emit_probs,
    trans_probs,
    beam_width=5  # Keep top 5 hypotheses at each step
)

print(f"Beam search path: {beam_path}")
print(f"Beam search score: {beam_score}")
```

## Algorithm Details

### Viterbi Algorithm

The Viterbi algorithm finds the most likely sequence using dynamic programming:

1. **Initialization**: For each state s at t=0, compute v_0(s) = P(s) * P(obs_0|s)
2. **Recursion**: For each subsequent time step t and state s:
   - v_t(s) = P(obs_t|s) * max_s'(v_t-1(s') * P(s|s'))
   - Track the best previous state for backtracking
3. **Termination**: Find the state with highest probability at final step
4. **Backtrack**: Follow parent pointers to reconstruct path

Time Complexity: O(n * |S|^2) where n is sequence length, |S| is number of states
Space Complexity: O(n * |S|)

### Beam Search

A memory-efficient variant that keeps only k best paths:

1. **Initialization**: Rank all states by their initial probability, keep top k
2. **Recursion**: For each step, extend the k active paths with all possible next states, then trim to top k
3. **Termination**: Return the best complete path

Time Complexity: O(n * k * |S|) where k is beam width
Space Complexity: O(k * n) instead of O(n * |S|)

## Features

- Log-space computation for numerical stability
- Support for arbitrary probability distributions
- Comprehensive error handling
- Full unit test coverage (12 test cases)
- Type hints throughout
- Clean, documented code

## Running Tests

```bash
python -m pytest tests.py -v
```

Or with unittest:

```bash
python -m unittest tests.py
```

## Design Decisions

1. **Log-Space by Default**: Probabilities often become vanishingly small; log-space avoids underflow
2. **Separate Viterbi/Beam**: Users can choose algorithm based on their constraints
3. **Flexible Input**: Arbitrary Python dictionaries for probabilities allows easy extension
4. **No External Dependencies**: Pure Python for maximum portability

## Performance Notes

- For sequences < 1000 observations: Standard Viterbi is usually fine
- For |S| (state count) > 100: Consider beam search to reduce memory
- Log-space computation adds about 10-15% overhead but is essential for stability

## Common Use Cases

1. **Part-of-Speech Tagging**: Sequence of words -> sequence of POS tags
2. **Named Entity Recognition**: Text -> entity labels (PERSON, ORG, LOC, etc.)
3. **Speech Recognition**: Acoustic features -> phonemes or words
4. **Gene Prediction**: DNA sequences -> coding/non-coding regions
5. **Weather Prediction**: Observations -> hidden weather states

## References

- Viterbi, A. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm
- Wikipedia: Viterbi Algorithm (https://en.wikipedia.org/wiki/Viterbi_algorithm)
- Speech and Language Processing by Jurafsky & Martin (Chapter on HMMs)

## License

MIT License

## Author

Mohammad Hossinzehi

Built as part of a series of production-grade algorithm implementations. Designed for both learning and practical use.

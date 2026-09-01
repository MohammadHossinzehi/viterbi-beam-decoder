"""
Viterbi decoder with beam search for optimal sequence labeling.

Implements dynamic programming algorithms for finding the most likely sequence
of hidden states given observations, with support for both standard Viterbi
and beam search variants for memory efficiency.
"""

import heapq
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass(order=True)
class BeamState:
    """Represents a state in the beam search."""
    score: float
    path: Tuple = None
    state_seq: Tuple = None
    
    def __lt__(self, other):
        return self.score > other.score  # Max heap


class ViterbiDecoder:
    """
    Viterbi decoder for finding optimal sequences in HMMs and sequence models.
    
    Supports:
    - Standard Viterbi algorithm (O(n*|S|^2) time, O(n*|S|) space)
    - Beam search variant (reduced memory at cost of optimality)
    - Log-space computation to avoid underflow
    - Custom transition and emission probabilities
    """
    
    def __init__(self, states: List[str], log_probs: bool = True):
        """
        Initialize decoder.
        
        Args:
            states: List of possible hidden states
            log_probs: Whether to work in log space (recommended for numerical stability)
        """
        self.states = states
        self.state_idx = {s: i for i, s in enumerate(states)}
        self.log_probs = log_probs
    
    def viterbi(
        self,
        obs_seq: List,
        start_probs: Dict[str, float],
        emit_probs: Dict[Tuple[str, any], float],
        trans_probs: Dict[Tuple[str, str], float]
    ) -> Tuple[List[str], float]:
        """
        Find most likely sequence of states using standard Viterbi algorithm.
        
        Args:
            obs_seq: Sequence of observations
            start_probs: P(state) at t=0
            emit_probs: P(obs | state)
            trans_probs: P(next_state | current_state)
            
        Returns:
            (most_likely_path, path_score)
        """
        n = len(obs_seq)
        if n == 0:
            return [], 0.0
        
        # Initialize DP table: dp[t][s] = (best_score, best_prev_state)
        dp = [[(-float('inf'), None) for _ in self.states] for _ in range(n)]
        
        # Base case: t=0
        for s in self.states:
            emit_key = (s, obs_seq[0])
            emit_prob = emit_probs.get(emit_key, 1e-10)
            start_prob = start_probs.get(s, 1e-10)
            
            if self.log_probs:
                score = math.log(start_prob) + math.log(emit_prob)
            else:
                score = start_prob * emit_prob
            
            dp[0][self.state_idx[s]] = (score, None)
        
        # Fill DP table
        for t in range(1, n):
            for curr_s in self.states:
                best_score = -float('inf')
                best_prev = None
                
                for prev_s in self.states:
                    prev_score, _ = dp[t-1][self.state_idx[prev_s]]
                    trans_key = (prev_s, curr_s)
                    trans_prob = trans_probs.get(trans_key, 1e-10)
                    
                    if self.log_probs:
                        score = prev_score + math.log(trans_prob)
                    else:
                        score = prev_score * trans_prob
                    
                    if score > best_score:
                        best_score = score
                        best_prev = prev_s
                
                # Add emission probability
                emit_key = (curr_s, obs_seq[t])
                emit_prob = emit_probs.get(emit_key, 1e-10)
                
                if self.log_probs:
                    best_score += math.log(emit_prob)
                else:
                    best_score *= emit_prob
                
                dp[t][self.state_idx[curr_s]] = (best_score, best_prev)
        
        # Backtrack to find path
        path = []
        best_last_state = None
        best_last_score = -float('inf')
        
        for s in self.states:
            score, _ = dp[n-1][self.state_idx[s]]
            if score > best_last_score:
                best_last_score = score
                best_last_state = s
        
        # Reconstruct path
        current = best_last_state
        for t in range(n-1, -1, -1):
            path.append(current)
            _, prev_state = dp[t][self.state_idx[current]]
            if prev_state is not None:
                current = prev_state
        
        path.reverse()
        return path, best_last_score
    
    def beam_search(
        self,
        obs_seq: List,
        start_probs: Dict[str, float],
        emit_probs: Dict[Tuple[str, any], float],
        trans_probs: Dict[Tuple[str, str], float],
        beam_width: int = 3
    ) -> Tuple[List[str], float]:
        """
        Find likely sequence using beam search (memory-efficient alternative to Viterbi).
        
        Args:
            obs_seq: Sequence of observations
            start_probs: P(state) at t=0
            emit_probs: P(obs | state)
            trans_probs: P(next_state | current_state)
            beam_width: Number of hypotheses to keep at each step
            
        Returns:
            (most_likely_path, path_score)
        """
        if len(obs_seq) == 0:
            return [], 0.0
        
        # Initialize beam with start probabilities
        beam = []
        for s in self.states:
            start_prob = start_probs.get(s, 1e-10)
            emit_key = (s, obs_seq[0])
            emit_prob = emit_probs.get(emit_key, 1e-10)
            
            if self.log_probs:
                score = math.log(start_prob) + math.log(emit_prob)
            else:
                score = start_prob * emit_prob
            
            heapq.heappush(beam, BeamState(score=score, path=(s,), state_seq=(s,)))
        
        # Keep only top beam_width
        beam = heapq.nlargest(beam_width, beam)
        
        # Process each observation
        for t in range(1, len(obs_seq)):
            new_beam = []
            
            for state_obj in beam:
                for next_s in self.states:
                    trans_key = (state_obj.path[-1], next_s)
                    trans_prob = trans_probs.get(trans_key, 1e-10)
                    emit_key = (next_s, obs_seq[t])
                    emit_prob = emit_probs.get(emit_key, 1e-10)
                    
                    if self.log_probs:
                        score = state_obj.score + math.log(trans_prob) + math.log(emit_prob)
                    else:
                        score = state_obj.score * trans_prob * emit_prob
                    
                    new_path = state_obj.path + (next_s,)
                    new_state_seq = state_obj.state_seq + (next_s,)
                    heapq.heappush(new_beam, 
                        BeamState(score=score, path=new_path, state_seq=new_state_seq))
            
            # Trim to beam width
            beam = heapq.nlargest(beam_width, new_beam)
        
        if beam:
            best = beam[0]
            return list(best.path), best.score
        return [], 0.0


# Example usage and testing
if __name__ == "__main__":
    # Simple weather HMM
    states = ["Sunny", "Rainy"]
    start_probs = {"Sunny": 0.8, "Rainy": 0.2}
    
    # Transition probabilities
    trans_probs = {
        ("Sunny", "Sunny"): 0.7,
        ("Sunny", "Rainy"): 0.3,
        ("Rainy", "Sunny"): 0.4,
        ("Rainy", "Rainy"): 0.6
    }
    
    # Emission probabilities (hidden state -> observation)
    emit_probs = {
        ("Sunny", "happy"): 0.8,
        ("Sunny", "grumpy"): 0.2,
        ("Rainy", "happy"): 0.4,
        ("Rainy", "grumpy"): 0.6
    }
    
    observations = ["happy", "grumpy", "grumpy", "happy", "happy"]
    
    decoder = ViterbiDecoder(states, log_probs=True)
    
    # Test Viterbi
    path, score = decoder.viterbi(observations, start_probs, emit_probs, trans_probs)
    print(f"Viterbi path: {path}")
    print(f"Score: {score}")
    
    # Test beam search
    beam_path, beam_score = decoder.beam_search(observations, start_probs, emit_probs, 
                                                  trans_probs, beam_width=2)
    print(f"\nBeam search path: {beam_path}")
    print(f"Score: {beam_score}")

"""
Unit tests for Viterbi decoder implementation.

Tests verify:
- Correctness of Viterbi algorithm
- Beam search approximation quality
- Edge cases (empty sequences, single state)
- Log-space numerical stability
"""

import unittest
import math
from viterbi_decoder import ViterbiDecoder


class TestViterbiDecoder(unittest.TestCase):
    """Test cases for ViterbiDecoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.states = ["Sunny", "Rainy"]
        self.start_probs = {"Sunny": 0.8, "Rainy": 0.2}
        
        self.trans_probs = {
            ("Sunny", "Sunny"): 0.7,
            ("Sunny", "Rainy"): 0.3,
            ("Rainy", "Sunny"): 0.4,
            ("Rainy", "Rainy"): 0.6
        }
        
        self.emit_probs = {
            ("Sunny", "happy"): 0.8,
            ("Sunny", "grumpy"): 0.2,
            ("Rainy", "happy"): 0.4,
            ("Rainy", "grumpy"): 0.6
        }
        
        self.decoder = ViterbiDecoder(self.states, log_probs=True)
    
    def test_empty_sequence(self):
        """Test that empty sequence returns empty path."""
        path, score = self.decoder.viterbi([], self.start_probs, self.emit_probs, self.trans_probs)
        self.assertEqual(path, [])
        self.assertEqual(score, 0.0)
    
    def test_single_observation(self):
        """Test decoding with single observation."""
        obs = ["happy"]
        path, score = self.decoder.viterbi(obs, self.start_probs, self.emit_probs, self.trans_probs)
        
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], "Sunny")  # Most likely given happy observation
        self.assertIsInstance(score, float)
        self.assertGreater(score, -float('inf'))
    
    def test_sequence_length_matches(self):
        """Test that decoded path length matches observation sequence."""
        obs = ["happy", "grumpy", "grumpy", "happy"]
        path, score = self.decoder.viterbi(obs, self.start_probs, self.emit_probs, self.trans_probs)
        
        self.assertEqual(len(path), len(obs))
        self.assertTrue(all(s in self.states for s in path))
    
    def test_log_space_numerical_stability(self):
        """Test that log-space computation is numerically stable."""
        # Create sequence with very small probabilities
        small_trans = {k: v * 0.01 for k, v in self.trans_probs.items()}
        small_emit = {k: v * 0.01 for k, v in self.emit_probs.items()}
        small_start = {k: v * 0.01 for k, v in self.start_probs.items()}
        
        obs = ["happy"] * 20  # Long sequence
        
        # Log-space should not underflow
        decoder_log = ViterbiDecoder(self.states, log_probs=True)
        path_log, score_log = decoder_log.viterbi(obs, small_start, small_emit, small_trans)
        
        self.assertEqual(len(path_log), len(obs))
        self.assertNotEqual(score_log, -float('inf'))
        self.assertFalse(math.isnan(score_log))
    
    def test_beam_search_path_quality(self):
        """Test that beam search returns reasonable paths."""
        obs = ["happy", "grumpy", "grumpy", "happy", "happy"]
        
        # Get Viterbi solution
        viterbi_path, viterbi_score = self.decoder.viterbi(obs, self.start_probs, 
                                                             self.emit_probs, self.trans_probs)
        
        # Get beam search solution (beam width = 2, should approximate Viterbi)
        beam_path, beam_score = self.decoder.beam_search(obs, self.start_probs, 
                                                           self.emit_probs, self.trans_probs, 
                                                           beam_width=2)
        
        self.assertEqual(len(beam_path), len(obs))
        # Beam search score should be close to Viterbi (within reason)
        # Allow for approximation difference
        self.assertLessEqual(viterbi_score - beam_score, 1.0)
    
    def test_beam_search_vs_viterbi(self):
        """Test that increasing beam width approaches Viterbi solution."""
        obs = ["happy", "grumpy", "happy"]
        
        viterbi_path, viterbi_score = self.decoder.viterbi(obs, self.start_probs, 
                                                             self.emit_probs, self.trans_probs)
        
        # With beam_width = len(states), should get Viterbi result
        beam_path_wide, beam_score_wide = self.decoder.beam_search(obs, self.start_probs, 
                                                                     self.emit_probs, self.trans_probs, 
                                                                     beam_width=len(self.states))
        
        self.assertEqual(beam_path_wide, viterbi_path)
        self.assertAlmostEqual(beam_score_wide, viterbi_score, places=5)
    
    def test_state_constraints(self):
        """Test that decoded states are valid."""
        obs = ["happy", "grumpy", "happy", "grumpy", "grumpy"]
        path, score = self.decoder.viterbi(obs, self.start_probs, self.emit_probs, self.trans_probs)
        
        for state in path:
            self.assertIn(state, self.states)
    
    def test_score_monotonicity_viterbi(self):
        """Test that Viterbi score is monotonic with sequence length."""
        obs_short = ["happy"]
        obs_long = ["happy", "happy", "happy"]
        
        path1, score1 = self.decoder.viterbi(obs_short, self.start_probs, self.emit_probs, self.trans_probs)
        path2, score2 = self.decoder.viterbi(obs_long, self.start_probs, self.emit_probs, self.trans_probs)
        
        # Longer sequence should have lower or equal probability (less likely)
        self.assertLessEqual(score2, score1)
    
    def test_deterministic_results(self):
        """Test that decoding is deterministic."""
        obs = ["happy", "grumpy", "happy"]
        
        # Run twice
        path1, score1 = self.decoder.viterbi(obs, self.start_probs, self.emit_probs, self.trans_probs)
        path2, score2 = self.decoder.viterbi(obs, self.start_probs, self.emit_probs, self.trans_probs)
        
        self.assertEqual(path1, path2)
        self.assertEqual(score1, score2)


class TestBeamSearch(unittest.TestCase):
    """Specific tests for beam search variant."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.states = ["A", "B", "C"]
        self.start_probs = {"A": 0.5, "B": 0.3, "C": 0.2}
        
        self.trans_probs = {
            ("A", "A"): 0.6, ("A", "B"): 0.2, ("A", "C"): 0.2,
            ("B", "A"): 0.2, ("B", "B"): 0.6, ("B", "C"): 0.2,
            ("C", "A"): 0.2, ("C", "B"): 0.2, ("C", "C"): 0.6
        }
        
        self.emit_probs = {
            ("A", "x"): 0.7, ("A", "y"): 0.3,
            ("B", "x"): 0.4, ("B", "y"): 0.6,
            ("C", "x"): 0.3, ("C", "y"): 0.7
        }
        
        self.decoder = ViterbiDecoder(self.states, log_probs=True)
    
    def test_beam_width_one(self):
        """Test beam search with width 1 (greedy)."""
        obs = ["x", "y", "x"]
        path, score = self.decoder.beam_search(obs, self.start_probs, self.emit_probs, 
                                                self.trans_probs, beam_width=1)
        
        self.assertEqual(len(path), len(obs))
        self.assertTrue(all(s in self.states for s in path))
    
    def test_beam_width_larger_than_states(self):
        """Test beam search with width larger than number of states."""
        obs = ["x", "y", "x"]
        path, score = self.decoder.beam_search(obs, self.start_probs, self.emit_probs, 
                                                self.trans_probs, beam_width=10)
        
        self.assertEqual(len(path), len(obs))


if __name__ == "__main__":
    unittest.main()

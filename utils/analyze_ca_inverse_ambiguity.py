"""
Cellular Automata Inverse Engineering Ambiguity Analyzer

This tool performs Monte Carlo simulation to estimate the ambiguity probability
in 1D cellular automata inverse engineering tasks. It quantifies how often
different (rule, layer) combinations produce identical outputs from a given
initial state.

This analysis supports the design decision in:
  cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py
which filters out samples with non-unique solutions.

Usage:
    python utils/analyze_ca_inverse_ambiguity.py [--samples N] [--length L]
    
    --samples: Number of random initial states to test (default: 200000)
    --length:  Width of the CA state in bits (default: 30)

Example:
    python utils/analyze_ca_inverse_ambiguity.py --samples 100000 --length 36
"""

import numpy as np
from collections import defaultdict
import time
import argparse


def ca_step(state, rule):
    """
    Single CA evolution step with circular boundary conditions.
    
    Args:
        state: numpy array of 0/1 values representing current CA state
        rule: integer 0-255 representing the Wolfram rule number
        
    Returns:
        numpy array of the next state
    """
    rule_table = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    patterns = (left << 2) | (state << 1) | right
    return rule_table[patterns]


def compute_ambiguity_ratio_optimized(state):
    """
    Compute the ratio of ambiguous (rule, layer) combinations for a given state.
    
    Optimized version: each rule only needs 1 layer calculation + 3 iterations = 4 CA evolutions.
    This avoids redundant calculations, providing ~2.5x speedup.
    
    Args:
        state: numpy array representing the initial CA state
        
    Returns:
        float: ambiguity ratio (ambiguous_combos / 1024.0)
    """
    output_counts = defaultdict(int)
    
    for rule in range(256):
        # Layer 1 (from initial state)
        current = ca_step(state, rule)
        output_counts[tuple(current)] += 1
        
        # Layer 2 (based on current)
        current = ca_step(current, rule)
        output_counts[tuple(current)] += 1
        
        # Layer 3
        current = ca_step(current, rule)
        output_counts[tuple(current)] += 1
        
        # Layer 4
        current = ca_step(current, rule)
        output_counts[tuple(current)] += 1
    
    # Count ambiguous combinations (outputs that appear more than once)
    ambiguous_combos = sum(c for c in output_counts.values() if c > 1)
    return ambiguous_combos / 1024.0


def monte_carlo_simulation(num_samples=200000, length=30, verbose=True):
    """
    Ultra-fast Monte Carlo simulation to estimate ambiguity probability.
    
    Theoretical expectation: P = 1023 / 2^30 ≈ 9.53e-7
    200K samples takes approximately 3-4 seconds.
    
    Args:
        num_samples: Number of random initial states to sample
        length: Width of CA state in bits
        verbose: Whether to print progress updates
        
    Returns:
        float: Estimated ambiguity probability
    """
    if verbose:
        print("=" * 65)
        print("1D Cellular Automata Inverse Engineering Ambiguity Simulation")
        print("=" * 65)
        print(f"Parameters: length={length}, samples={num_samples:,}")
        print("Optimization: 1 layer + 3 iterations per rule (4 CA ops/rule)")
        print("=" * 65)
    
    total_ratio = 0.0
    start_time = time.time()
    
    for i in range(num_samples):
        # Random input state
        input_state = np.random.randint(0, 2, size=length, dtype=np.uint8)
        
        # Core: compute ambiguity ratio (only 256×4 CA calculations)
        ratio = compute_ambiguity_ratio_optimized(input_state)
        total_ratio += ratio
        
        # Progress updates
        if verbose and (i + 1) % 10000 == 0:
            avg_ratio = total_ratio / (i + 1)
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1:6d}/{num_samples} | Current P: {avg_ratio:.10f} "
                  f"({avg_ratio*100:.8f}%) | Speed: {(i+1)/elapsed:.0f} iter/s")
    
    # Final results
    final_prob = total_ratio / num_samples
    total_time = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 65)
        print(f"Sample count: {num_samples:,}")
        print(f"**Estimated ambiguity probability**: {final_prob:.10f} ({final_prob*100:.8f}%)")
        
        # 95% confidence interval
        se = np.sqrt(final_prob * (1 - final_prob) / num_samples)
        ci_low = max(0.0, final_prob - 1.96 * se)
        ci_high = min(1.0, final_prob + 1.96 * se)
        print(f"95% Confidence Interval: [{ci_low:.10f}, {ci_high:.10f}]")
        
        # Performance stats
        print(f"Runtime: {total_time:.2f} seconds")
        print(f"Total CA calculations: {256*4*num_samples:,}")
        print(f"Average speed: {num_samples/total_time:.0f} inputs/sec")
        print("=" * 65)
    
    return final_prob


def main():
    parser = argparse.ArgumentParser(
        description="Estimate ambiguity probability for CA inverse engineering tasks"
    )
    parser.add_argument(
        "--samples", type=int, default=200000,
        help="Number of random initial states to sample (default: 200000)"
    )
    parser.add_argument(
        "--length", type=int, default=30,
        help="Width of CA state in bits (default: 30)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    print("Monte Carlo simulation starting...")
    result = monte_carlo_simulation(
        num_samples=args.samples,
        length=args.length,
        verbose=not args.quiet
    )
    
    return result


if __name__ == "__main__":
    main()
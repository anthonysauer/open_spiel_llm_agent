import math
from mcts_ttt_rewards import *

def main():
    # =============================================================================
    # START STATE
    # =============================================================================

    # Test correct start state
    correct_start_state_rewards = mcts_individual_rewards(
        ["<reasoning>Start state: [x(0,0); o(2,0); x(0,2); o(2,2); x(2,1)]</reasoning>"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]],
    )[0]
    assert correct_start_state_rewards == [1.0]
    print("correct_start_state_rewards: PASSED")

    # Test incorrect start state
    incorrect_start_state_rewards = mcts_individual_rewards(
        ["<reasoning>Start state: [x(1,0); o(2,0); x(0,2); o(2,2); x(2,1)]</reasoning>"],
        [["x(0,1)", "o(0,2)", "x(1,0)", "o(1,1)", "x(2,2)", "o(2,1)"]],
    )[0]
    assert incorrect_start_state_rewards == [-1.0]
    print("incorrect_start_state_rewards: PASSED")


    # =============================================================================
    # VALID MOVES
    # =============================================================================

    # Test that valid moves are correctly rewarded in exploration
    explores_valid_moves = mcts_individual_rewards(
        ["Exploring move sequence: [o(1,1)]\nEvaluating state with random playouts:\n[x(1,2); o(0,1); x(1,0)] -> draw\n[x(0,1)] -> x wins\n[x(1,0); o(1,2); x(0,1)] -> x wins\n[x(1,2); o(1,0); x(0,1)] -> x wins\n\nExploring move sequence: [o(0,1)]\nEvaluating state with random playouts:\n[x(1,0); o(1,2); x(1,1)] -> draw\n[x(1,2); o(1,0); x(1,1)] -> draw\n[x(1,1); o(1,0); x(1,2)] -> draw\n[x(1,2); o(1,1); x(1,0)] -> draw\n\nExploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,2); o(1,1); x(0,1)] -> x wins\n[x(1,2); o(0,1); x(1,1)] -> draw\n[x(1,1); o(1,2); x(0,1)] -> x wins\n[x(1,1); o(0,1); x(1,2)] -> draw\n\n"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]]
    )[1]
    assert math.isclose(explores_valid_moves[0], 0.3)
    print("explores_valid_moves: PASSED")

    # Test that invalid moves are not rewarded in exploration
    explores_invalid_moves = mcts_individual_rewards(
        ["Exploring move sequence: [o(2,1)]\nEvaluating state with random playouts:\n[x(1,2); o(0,1); x(1,0)] -> draw\n[x(0,1)] -> x wins\n[x(1,0); o(1,2); x(0,1)] -> x wins\n[x(1,2); o(1,0); x(0,1)] -> x wins\n\nExploring move sequence: [o(3,1)]\nEvaluating state with random playouts:\n[x(1,0); o(1,2); x(1,1)] -> draw\n[x(1,2); o(1,0); x(1,1)] -> draw\n[x(1,1); o(1,0); x(1,2)] -> draw\n[x(1,2); o(1,1); x(1,0)] -> draw\n\nExploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,2); o(1,1); x(0,1)] -> x wins\n[x(1,2); o(0,1); x(1,1)] -> draw\n[x(1,1); o(1,2); x(0,1)] -> x wins\n[x(1,1); o(0,1); x(1,2)] -> draw\n\n"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]]
    )[1]
    assert math.isclose(explores_invalid_moves[0], -0.1)
    print("explores_invalid_moves: PASSED")

    # Test that valid moves are correctly rewarded in playouts
    playout_valid_moves = mcts_individual_rewards(
        ["Exploring move sequence: [o(1,1)]\nEvaluating state with random playouts:\n[x(1,2); o(0,1); x(1,0)] -> draw\n[x(0,1)] -> x wins\n[x(1,0); o(1,2); x(0,1)] -> x wins\n[x(1,2); o(1,0); x(0,1)] -> x wins\n\nExploring move sequence: [o(0,1)]\nEvaluating state with random playouts:\n[x(1,0); o(1,2); x(1,1)] -> draw\n[x(1,2); o(1,0); x(1,1)] -> draw\n[x(1,1); o(1,0); x(1,2)] -> draw\n[x(1,2); o(1,1); x(1,0)] -> draw\n\nExploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,2); o(1,1); x(0,1)] -> x wins\n[x(1,2); o(0,1); x(1,1)] -> draw\n[x(1,1); o(1,2); x(0,1)] -> x wins\n[x(1,1); o(0,1); x(1,2)] -> draw\n\n"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]]
    )[3]
    assert math.isclose(playout_valid_moves[0], 1.2)
    print("playout_valid_moves: PASSED")

    # Test that invalid moves are not rewarded in playouts
    playout_invalid_moves = mcts_individual_rewards(
        ["Exploring move sequence: [o(2,1)]\nEvaluating state with random playouts:\n[x(1,2); o(0,1); x(1,0)] -> draw\n[x(0,1)] -> x wins\n[x(1,0); o(1,2); x(0,1)] -> x wins\n[x(1,2); o(1,0); x(0,1)] -> x wins\n\nExploring move sequence: [o(0,1)]\nEvaluating state with random playouts:\n[x(0,1); o(1,2); x(1,1)] -> draw\n[x(1,2); o(1,0); x(2,1)] -> draw\n[x(1,1); o(1,0); x(1,2)] -> draw\n[x(1,2); o(1,1); x(1,0)] -> draw\n\nExploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,2); o(1,1); x(0,1)] -> x wins\n[x(1,2); o(0,1); x(1,1)] -> draw\n[x(1,1); o(1,2); x(0,1)] -> x wins\n[x(1,1); o(3,1); x(1,2)] -> draw\n\n"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]]
    )[3]
    assert math.isclose(playout_invalid_moves[0], -0.2)
    print("playout_invalid_moves: PASSED")


    # =============================================================================
    # RECOGNIZES TERMINAL STATES
    # =============================================================================

    # Test when terminal states are correctly recognized in exploration
    explores_correct_terminal = mcts_individual_rewards(
        ["Exploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,1)] -> x wins\n\nExploring move sequence: [o(1,1)]\no wins\n\n"],
        [["x(0,0)", "o(0,1)", "x(1,2)", "o(0,2)", "x(2,1)", "o(2,0)", "x(2,2)"]]
    )[2]
    assert math.isclose(explores_correct_terminal[0], 0.1)
    print("explores_correct_terminal: PASSED")

    # Test when terminal states are incorrectly recognized in exploration
    explores_incorrect_terminal = mcts_individual_rewards(
        ["Exploring move sequence: [o(1,1)]\nEvaluating state with random playouts:\n[x(1,1)] -> x wins\n\nExploring move sequence: [o(1,0)]\no wins\n\nExploring move sequence: [o(1,1)]\ndraw\n\n"],
        [["x(0,0)", "o(0,1)", "x(1,2)", "o(0,2)", "x(2,1)", "o(2,0)", "x(2,2)"]]
    )[2]
    assert math.isclose(explores_incorrect_terminal[0], -0.3)
    print("explores_incorrect_terminal: PASSED")

    # Test when terminal states are correctly recognized in playouts
    playout_correct_terminal = mcts_individual_rewards(
        ["Exploring move sequence: [o(1,1)]\nEvaluating state with random playouts:\n[x(1,2); o(0,1); x(1,0)] -> draw\n[x(0,1)] -> x wins\n[x(1,0); o(1,2); x(0,1)] -> x wins\n[x(1,2); o(1,0); x(0,1)] -> x wins\n\nExploring move sequence: [o(0,1)]\nEvaluating state with random playouts:\n[x(1,0); o(1,2); x(1,1)] -> draw\n[x(1,2); o(1,0); x(1,1)] -> draw\n[x(1,1); o(1,0); x(1,2)] -> draw\n[x(1,2); o(1,1); x(1,0)] -> draw\n\nExploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,2); o(1,1); x(0,1)] -> x wins\n[x(1,2); o(0,1); x(1,1)] -> draw\n[x(1,1); o(1,2); x(0,1)] -> x wins\n[x(1,1); o(0,1); x(1,2)] -> draw\n\n"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]]
    )[4]
    assert math.isclose(playout_correct_terminal[0], 1.2)
    print("playout_correct_terminal: PASSED")

    # Test when terminal states are incorrectly recognized in playouts
    playout_incorrect_terminal = mcts_individual_rewards(
        ["Exploring move sequence: [o(1,1)]\nEvaluating state with random playouts:\n[x(1,2); o(0,1); x(1,0)] -> x wins\n[x(0,1); o(0,1)] -> x wins\n[x(1,0); o(1,2)] -> x wins\n[x(1,2); o(1,0); x(0,1)] -> x wins\n\nExploring move sequence: [o(0,1)]\nEvaluating state with random playouts:\n[x(1,0); o(1,2); x(1,1)] -> draw\n[x(1,2); o(1,0); x(1,1)] -> draw\n[x(1,1); o(1,0); x(1,2)] -> draw\n[x(1,2); o(1,1); x(1,0)] -> draw\n\nExploring move sequence: [o(1,0)]\nEvaluating state with random playouts:\n[x(1,2); o(1,1); x(0,1)] -> draw\n[x(1,2); o(0,1); x(1,1)] -> draw\n[x(1,1); o(1,2); x(0,1)] -> x wins\n[x(1,1); o(0,1); x(1,2)] -> draw\n\n"],
        [["x(0,0)", "o(2,0)", "x(0,2)", "o(2,2)", "x(2,1)"]]
    )[4]
    assert math.isclose(playout_incorrect_terminal[0], 0.4)
    print("playout_incorrect_terminal: PASSED")


if __name__ == "__main__":
    main()
    print("All passed")

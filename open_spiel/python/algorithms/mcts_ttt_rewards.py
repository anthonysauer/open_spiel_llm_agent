import numpy as np
import pyspiel
import re
import wandb

from transformers.utils import logging

# Set-up logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# RegEx for parsing reasoning
single_move_pattern = r"[xo]\(\d\s*,\d\)"
state_pattern = r"\[(?:" + single_move_pattern + r"(?:;\s*)?)+\]"
outcome_pattern = r"(?:draw)|(?:[xo] wins?)"
exploration_pattern = r"[Ee]xploring.*" + state_pattern + r"\n(?:" + outcome_pattern + r"\n)?"
playout_pattern = exploration_pattern + r".*playouts.*\n(?:" + state_pattern + r"\s*->\s*(?:" + outcome_pattern + r")\n+)+"
playout_pattern_with_groups = "(" + exploration_pattern + ")" + r".*playouts.*\n((?:" + state_pattern + r"\s*->\s*(?:" + outcome_pattern + r")\n+)+)"


def extract_xml_reasoning(text: str) -> str:
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# # Strict format of overall response
# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r".*?<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n.*?"
#     matches = [re.match(pattern, c) for c in completions]
#     return [0.5 if match else 0.0 for match in matches]


# Soft format of overall response
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>\n[\S\s]*<\/reasoning>\n<answer>\n.*?\n<\/answer>\n"
    matches = [re.search(pattern, c) for c in completions]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# Correct number of xml tags
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]


# # Strict reasoning formatting -- nearly exact match with training data format
# def strict_reasoning_format_reward_func(completions, **kwargs) -> list[float]:
#     extracted_reasonings = [extract_xml_reasoning(c) for c in completions]
#     pattern = r".*?Start state:\s*\n([xo.]{3}\n){3}Evaluating state with random playout:\s*\n([xo]\([0-2],[0-2]\)\n([xo.]{3}\n){3})+(([xo] wins)|(draw)) in random playout\nUpdating move rewards:\s*\n\n(Exploring move sequence: \[('[xo]\([0-2],[0-2]\)'(, )?)+\]\nResulting state:\s*\n([xo.]{3}\n){3}((draw\n)|([xo] wins\n))?(Evaluating state with random playout:\s*\n([xo]\([0-2],[0-2]\)\n([xo.]{3}\n){3})+(([xo] wins)|(draw)) in random playout\n)?Updating move rewards:\s*\n((\[('[xo]\([0-2],[0-2]\)'(, )?)+\]: -?\d.\d \+ -?\d.\d = -?\d.\d\n)|(All possible moves starting from move sequence \[('[xo]\([0-2],[0-2]\)'(, )?)+\] have been solved\. This move sequence has a maximum reward of -?\d\.\d for x\n)|(\[('[xo]\([0-2],[0-2]\)'(, )?)+\] is a winning move sequence for [xo]\n))+\n)+(((Explored moves at best resulted in a draw for [xo].\nChoosing move that results in draw that was explored the most: [xo]\([0-2],[0-2]\)\n)|(Choosing move that is proven to win for [xo]: [xo]\([0-2],[0-2]\)\n))|(No explored moves are a winning move for [xo].\nChoosing move with highest reward that is not a losing move for [xo]: [xo]\([0-2],[0-2]\)\n)).*?"
#     return [1.0 if re.match(pattern, r) else 0.0 for r in extracted_reasonings]


# Soft reasoning formatting
def soft_reasoning_format_reward_func(completions, **kwargs) -> list[float]:
    extracted_reasonings = [extract_xml_reasoning(c) for c in completions]
    pattern = r"[sS]tart.*\[(?:[xo]\([0-2]\s*,[0-2]\)(?:;\s*)?)+\]\n*(?:[Ee]xploring.*\[(?:[xo]\([0-2]\s*,[0-2]\)(?:;\s*)?)+\]\n(?:(?:(?:draw)|(?:[xo] wins?)\n)|(?:.*playouts.*\n(?:\[(?:[xo]\([0-2]\s*,[0-2]\)(?:;\s*)?)+\]\s*->\s*(?:(?:draw)|(?:[xo] wins?))\n)+\s*))(?:[\S\s])??)+[\S\s]*"
    return [1.0 if re.search(pattern, r) else 0.0 for r in extracted_reasonings]


# Final move is in the strict format, e.g. x(0,1)
def move_format_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(c) for c in completions]
    pattern = r"^[xo]\([0-2],[0-2]\)$"
    return [0.5 if re.match(pattern, r) else 0.0 for r in extracted_responses]


def get_action(state, action_str):
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None


def get_start_state(current_moves):
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    for action_str in current_moves:
        state.apply_action(get_action(state, action_str))
    return state


# Simulates Tic-Tac-Toe game using the given move sequences and start state,
# and provides reward counts for the number of valid moves and terminal states
def count_valid_ttt_sequences(start_state, move_sequences, outcomes):
    valid_move_sequence_count = 0.0
    correct_terminal_count = 0.0

    for move_sequence, outcome in zip(move_sequences, outcomes):
        state = start_state.clone()
        move_sequence_valid = True
        true_outcome = None
        terminal_valid = True

        for i, move in enumerate(move_sequence):
            action = get_action(state, move)
            if action is None:
                move_sequence_valid = False
                break

            state.apply_action(action)

            if state.is_terminal():
                if i < (len(move_sequence) - 1):  # If a terminal state is reached but there are still moves remaining
                    terminal_valid = False
                else:
                    true_returns = state.returns()
                    if true_returns[0] == 0.0 and true_returns[1] == 0.0:
                        true_outcome = "draw"
                    elif true_returns[0] == 1.0:
                        true_outcome = "x wins"
                    else:
                        true_outcome = "o wins"

        if move_sequence_valid:
            valid_move_sequence_count += 0.1

            if terminal_valid:
                if true_outcome is None and outcome is None:
                    continue
                elif true_outcome == outcome:
                    correct_terminal_count += 0.1
                else:
                    correct_terminal_count -= 0.1
            else:
                correct_terminal_count -= 0.1
        else:
            valid_move_sequence_count -= 0.1

            if not terminal_valid:
                correct_terminal_count -= 0.1

    return valid_move_sequence_count, correct_terminal_count


# Correct start state based on previous moves
def mcts_individual_rewards(completions, current_moves):
    extracted_reasonings = [extract_xml_reasoning(c) for c in completions]
    start_states = [get_start_state(c) for c in current_moves]

    # Rewards for correct start state given previous moves
    start_state_matches = [re.search(state_pattern, r) for r in extracted_reasonings]
    start_state_rewards = [1.0 if (m is not None and m.group().strip() == "[" + "; ".join(c) + "]") else -1.0 for m, c
                           in zip(start_state_matches, current_moves)]

    # Rewards for exploring valid moves
    explores_valid_moves_rewards = []

    # Rewards for correctly identifying terminal vs non-terminal states
    explored_terminal_rewards = []

    exploration_lists = [re.findall(exploration_pattern, r) for r in extracted_reasonings]
    exploration_moves = []  # List (completions) of lists (explorations) of lists (move sequences)
    exploration_outcomes = []  # List (completions) of lists (terminal result of exploration or None if non-terminal)

    for explorations in exploration_lists:
        moves = []
        outcome_matches = []
        for e in explorations:
            moves.append(re.findall(single_move_pattern, e))
            outcome_matches.append(re.search(outcome_pattern, e))

        exploration_moves.append(moves)
        exploration_outcomes.append([m.group() if m is not None else None for m in outcome_matches])

    valid_exploration_counts = [
        count_valid_ttt_sequences(s, m, o)
        for s, m, o in zip(start_states, exploration_moves, exploration_outcomes)
    ]

    for count in valid_exploration_counts:
        explores_valid_moves_rewards.append(count[0])
        explored_terminal_rewards.append(count[1])

    # Rewards for adding valid moves to the playouts
    playout_valid_moves_rewards = []

    # Rewards for correctly identifying terminal vs non-terminal states in playouts
    playout_terminal_rewards = []

    playout_lists = [re.findall(playout_pattern, r) for r in extracted_reasonings]
    playout_moves = []  # List (completions) of lists (playouts) of lists (moves)
    playout_outcomes = []  # List (completions) of lists (terminal result of playout or None if non-terminal)

    for playouts in playout_lists:
        moves = []
        outcome_matches = []
        for playout in playouts:
            playout_match = re.search(playout_pattern_with_groups, playout)
            prev_moves = re.findall(single_move_pattern, playout_match.group(1))
            for p in playout_match.group(2).strip().splitlines():
                moves.append(prev_moves + re.findall(single_move_pattern, p))
                outcome_matches.append(re.search(outcome_pattern, p))

        playout_moves.append(moves)
        playout_outcomes.append([m.group() if m is not None else None for m in outcome_matches])

    valid_playout_counts = [
        count_valid_ttt_sequences(s, m, o)
        for s, m, o in zip(start_states, playout_moves, playout_outcomes)
    ]

    for count in valid_playout_counts:
        playout_valid_moves_rewards.append(count[0])
        playout_terminal_rewards.append(count[1])

    wandb.log({
        "start_state_reward_func": sum(start_state_rewards) / len(start_state_rewards),
        "explores_valid_moves_reward_func": sum(explores_valid_moves_rewards) / len(explores_valid_moves_rewards),
        "playouts_valid_moves_func": sum(playout_valid_moves_rewards) / len(playout_valid_moves_rewards),
        "explores_terminal_reward_func": sum(explored_terminal_rewards) / len(explored_terminal_rewards),
        "playouts_terminal_reward_func": sum(playout_terminal_rewards) / len(playout_terminal_rewards),
    })

    return start_state_rewards, explores_valid_moves_rewards, explored_terminal_rewards, playout_valid_moves_rewards, playout_terminal_rewards


# Sums together individual MCTS reward functions, to get final rewards
def mcts_reward_func(completions, current_moves, **kwargs) -> list[float]:
    return list(np.sum(np.array(mcts_individual_rewards(completions, current_moves)), axis=0))


# Final move is in list of optimal moves
def optimality_reward_func(completions, optimal_moves, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(c) for c in completions]
    return [4.0 if r in m else 0.0 for r, m in zip(extracted_responses, optimal_moves)]

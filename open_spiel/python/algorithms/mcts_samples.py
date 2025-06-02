# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generating Tic-Tac-Toe data using open_spiel.python.algorithms.mcts."""

import jsonlines
import math
import random

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import minimax
import pyspiel

UCT_C = math.sqrt(2)

num_examples = 5


def rollout(state, prev_actions, min_depth, max_depth, dedupe_states=False):
    if state.is_terminal() or len(prev_actions) > max_depth:
        return []

    states_with_actions = []
    visited = set()
    if len(prev_actions) >= min_depth:
        states_with_actions.append((state, prev_actions))
        visited.add(str(state))

    next_actions = state.legal_actions()
    if dedupe_states:
        random.shuffle(next_actions)

    for action in next_actions:
        working_state = state.clone()
        working_state.apply_action(action)

        working_actions = prev_actions.copy()
        working_actions.append(state.action_to_string(state.current_player(), action))

        next_states_with_actions = rollout(working_state, working_actions, min_depth, max_depth)

        if not dedupe_states:
            states_with_actions.extend(next_states_with_actions)
        else:
            for next_state, next_actions in next_states_with_actions:
                if str(next_state) not in visited:
                    states_with_actions.append((next_state, next_actions))
                    visited.add(str(next_state))

    return states_with_actions


def get_optimal_moves(game, state):
    optimal_outcome, _, optimal_actions = minimax.alpha_beta_search(game, state=state, return_all_actions=True)

    optimal_outcome_str = "draw"
    if optimal_outcome == 1:
        optimal_outcome_str = "win"
    elif optimal_outcome == -1:
        optimal_outcome_str = "loss"

    return [state.action_to_string(state.current_player(), action) for action in optimal_actions], optimal_outcome_str


def main():
    data = []
    game = pyspiel.load_game("tic_tac_toe")
    initial_state = game.new_initial_state()

    all_states_with_actions = rollout(initial_state, prev_actions=[], min_depth=2, max_depth=7, dedupe_states=True)
    random.shuffle(all_states_with_actions)

    bot = mcts.MCTSBot(
        game,
        UCT_C,
        max_simulations=10,
        solve=True,
        evaluator=mcts.RandomRolloutEvaluator(n_rollouts=1))

    for state, actions in all_states_with_actions:
        example = ""
        root, thought_data = bot.mcts_search(state, return_data=True)
        example += thought_data + "\n"

        best = root.best_child()
        best_move = state.action_to_string(best.player, best.action)
        player = 'x'
        if best.player == 1:
            player = 'o'

        if best.outcome is None or best.outcome[best.player] == -1:
            example += "No explored moves are a winning move for " + player + ".\n"
            example += "Choosing move with highest reward that is not a losing move for " + player + ": " + best_move + "\n"
        elif best.outcome[best.player] == 1:
            example += "Choosing move that is proven to win for " + player + ": " + best_move + "\n"
        else:
            example += "Explored moves at best resulted in a draw for " + player + ".\n"
            example += "Choosing move that results in draw that was explored the most: " + best_move + "\n"

        optimal_moves, optimal_outcome = get_optimal_moves(game, state)

        data.append({
            "current_moves": actions,
            "player": player,
            "prompt": "Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbols. The goal is to get 3 of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \n"
                      "Each move is represented by a string consisting of two parts: the current player ('x' or 'o') and the coordinate of their move (column, row), in that order. For instance, o(0,2) means that 'o' moves at the first column and the third row of the grid. \n"
                      "The current move sequence is: " + ", ".join(actions) + ". You, player " + player + ", will move next. \n"
                      "Think about your current situation, then choose the best next move by exploring the search space. \nYour output must be in the following format strictly:\n<reasoning>\nThe search space trace.\n</reasoning>\n<answer>The best move for player " + player + " (you), i.e., + " + player + "(column,row)</answer>\n",
            "reasoning": example,
            "answer": best_move,
            "optimal_moves": optimal_moves,
            "optimal_outcome": optimal_outcome,

            # "messages": [
            #     {"role": "system",
            #      "content": "You are a powerful gaming agent who can make proper decisions to beat the user in gaming tasks."},
            #     {"role": "user",
            #      "content": "Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbols. The goal is to get 3 of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \n"
            #                 "Each move is represented by a string consisting of two parts: the current player ('x' or 'o') and the coordinate of their move (column, row), in that order. For instance, o(0,2) means that 'o' moves at the first column and the third row of the grid. \n"
            #                 "The current move sequence is: " + ", ".join(
            #          actions) + ". You, player " + player + ", will move next. \n"
            #                                                 "Think about your current situation, then choose the best next move by exploring the search space. \nYour output must be in the following format strictly:\n\nThought:\nThe search space trace.\nAction:\nThe best move for player " + player + " (you), i.e., + " + player + "(column,row)"},
            #     {"role": "assistant",
            #      "content": "Thought:\n" + example + "\nAction:\n" + best_move}
            # ]
        })

    grpo_train_size = int(len(data) * 0.7)
    sft_train_size = int(len(data) * 0.2)
    train_size = grpo_train_size + sft_train_size
    test_size = len(data) - train_size
    print("Train split (GRPO): " + str(grpo_train_size))
    print("Train split (SFT): " + str(sft_train_size))
    print("Test split: " + str(test_size))

    with jsonlines.open("mcts_ttt_train_grpo.jsonl", "w") as writer:
        writer.write_all(data[:grpo_train_size])
    with jsonlines.open("mcts_ttt_train_sft.jsonl", "w") as writer:
        writer.write_all(data[grpo_train_size:train_size])
    with jsonlines.open("mcts_ttt_test.jsonl", "w") as writer:
        writer.write_all(data[train_size:])


if __name__ == "__main__":
    main()

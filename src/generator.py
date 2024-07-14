import numpy as np 
import pickle

class ClimbGenerator:
    def __init__(self, level):
        path = f'src/probabilities/{level}'
        start_path = f'{path}/start_prob.pkl'
        with open(start_path, 'rb') as file:
            self.start_probs = pickle.load(file)

        pair_path = f'{path}/pair_prob.pkl'
        with open(pair_path, 'rb') as file:
            self.pair_probs = pickle.load(file)

        cond_path = f'{path}/cond_prob.pkl'
        with open(cond_path, 'rb') as file:
            self.cond_probs = pickle.load(file)

        prob_path = f'{path}/prob_matrix.pkl'
        with open(prob_path, 'rb') as file:
            self.prob_matrix = pickle.load(file)

    def _sample_starting_hold(self, prob_starting_hold):
        holds = list(prob_starting_hold.keys())
        probabilites = list(prob_starting_hold.values())
        return str(np.random.choice(holds, p=probabilites))

    def _sample_second_starting_hold(self, hold, prob_with_pairs, conditional_second_starting_holds_info):
        if np.random.rand() < prob_with_pairs.get(hold, 0):
            second_holds = list(conditional_second_starting_holds_info[hold].keys())
            second_probabilities = list(conditional_second_starting_holds_info[hold].values())
            return np.random.choice(second_holds, p=second_probabilities)
        return None

    def _sample_next_hold(self, current_hold, probability_matrix):
        next_holds = probability_matrix.columns
        probabilities = probability_matrix.loc[current_hold]
        return np.random.choice(next_holds, p=probabilities)

    def _hold_indexer(self, hold_value):
        letter_dict = {
            'A': 0, 'B': 1, 'C':2, 'D': 3,
            'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'J': 9, 'K': 10
        }
        col = int(letter_dict[hold_value[0].upper()])
        row = int(hold_value[1:]) - 1

        index = row * 11 + col

        return index

    def generate_climb(self):  #, prob_starting_hold, prob_with_pairs, conditional_second_starting_holds_info, probability_matrix):
        climb = []
        labels = []

        # Sample the first starting hold
        first_hold = self._sample_starting_hold(self.start_probs)  # prob_starting_hold)
        climb.append(first_hold)
        labels.append('starting hold')

        # Determine if there's a second starting hold
        second_hold = self._sample_second_starting_hold(first_hold, self.pair_probs, self.cond_probs)  # prob_with_pairs, conditional_second_starting_holds_info)
        if second_hold:
            climb.append(second_hold)
            labels.append('starting hold')

        current_hold = climb[-1]
        current_row = int(current_hold[1:])  # Extract row number from hold description

        # Generate the rest of the climb
        while current_row < 18:
            next_hold = self._sample_next_hold(current_hold, self.prob_matrix)  # probability_matrix)
            climb.append(next_hold)
            labels.append('intermediate hold')
            current_hold = next_hold
            current_row = int(current_hold[1:])

        labels[-1] = 'finish hold'  # Label the last hold as the finish hold

        climb = [str(x) for x in climb]
        climb_idxs= [self._hold_indexer(x) for x in climb]
        climb_vector = np.zeros(198)

        for i in climb_idxs:
            climb_vector[i] += 1

        return climb, labels, climb_vector


if __name__ == '__main__':
    generator = ClimbGenerator('hard')
    _, _, climb_vector = generator.generate_climb()
    print(climb_vector)

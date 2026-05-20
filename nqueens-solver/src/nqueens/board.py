from typing import List
import random


class NQueensBoard:

    def __init__(self, n: int):

        self.n = n
        self.state = self.generate_random_state()

    def generate_random_state(self) -> List[int]:

        return [
            random.randint(0, self.n - 1)
            for _ in range(self.n)
        ]

    @staticmethod
    def count_conflicts(state: List[int]) -> int:

        conflicts = 0
        n = len(state)

        for i in range(n):

            for j in range(i + 1, n):

                same_row = state[i] == state[j]

                same_diagonal = (
                    abs(state[i] - state[j]) == abs(i - j)
                )

                if same_row or same_diagonal:
                    conflicts += 1

        return conflicts
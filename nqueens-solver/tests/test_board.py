from src.nqueens.board import NQueensBoard


def test_conflict_count():

    solution = [
        0, 4, 7, 5,
        2, 6, 1, 3
    ]

    conflicts = (
        NQueensBoard.count_conflicts(solution)
    )

    assert conflicts == 0


if __name__ == "__main__":

    test_conflict_count()

    print("All tests passed.")
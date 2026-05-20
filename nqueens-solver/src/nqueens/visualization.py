import matplotlib.pyplot as plt


def plot_board(state):

    n = len(state)

    fig, ax = plt.subplots(figsize=(8, 8))

    queen_size = max(8, 350 // n)

    for row in range(n):

        for col in range(n):

            color = (
                "#e4e0d9"
                if (row + col) % 2 == 0
                else "#60947a"
            )

            ax.add_patch(

                plt.Rectangle(
                    (col, n - row - 1),
                    1,
                    1,
                    color=color
                )
            )

            if state[col] == row:

                ax.text(
                    col + 0.5,
                    n - row - 0.5,
                    "♛",
                    ha="center",
                    va="center",
                    fontsize=queen_size,
                    color="black"
                )

    ax.set_xlim(0, n)

    ax.set_ylim(0, n)

    ax.set_aspect("equal")

    ax.axis("off")

    plt.title("N-Queens Solution")

    plt.tight_layout()

    plt.show()


def plot_history(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history, linewidth=2)

    plt.xlabel("Step")

    plt.ylabel("Conflicts")

    plt.title(
        "Optimization Progress"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()
# N-Queens Solver

A Python implementation of the classic N-Queens problem using local search optimization algorithms.

This project focuses on solving the problem using techniques such as Hill Climbing and Simulated Annealing while also visualizing the optimization process.

---

# Features

- Hill Climbing Algorithm
- Simulated Annealing Algorithm
- Random Restart Strategy
- Conflict Detection
- Chessboard Visualization
- Optimization Progress Plotting
- Runtime Measurement
- Command-Line Interface
- Basic Testing

---

# Problem Description

The N-Queens problem is a classical combinatorial optimization problem.

The objective is to place N queens on an N×N chessboard such that no two queens attack each other.

A queen can attack:

- Horizontally
- Vertically
- Diagonally

This project solves the problem using local search algorithms.

---

# Algorithms

## Hill Climbing

Hill Climbing repeatedly moves to a neighboring state with fewer conflicts until no better move can be found.

## Simulated Annealing

Simulated Annealing allows occasional non-optimal moves to help avoid local optima during the search process.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/vidasdti/PythonProjects.git
cd PythonProjects/nqueens-solver
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

Run Hill Climbing:

```bash
python -m src.nqueens.cli --algorithm hill --n 20
```

Run Simulated Annealing:

```bash
python -m src.nqueens.cli --algorithm annealing --n 20
```

---

# Example Output

```text
Algorithm: hill
Board Size: 20

Conflicts: 0
Restarts: 1
Steps: 24
Runtime: 0.018 seconds
```

---

# Visualization

The project generates:

- Chessboard visualization
- Conflict reduction graph

---

# Technologies Used

- Python
- Matplotlib
- argparse

---

# Testing

Run tests using:

```bash
python -m tests.test_board
```

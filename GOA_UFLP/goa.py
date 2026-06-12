import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time

# ====================== CONFIGURATION ======================

np.random.seed(42)

N_FACILITIES = 50
N_CUSTOMERS = 500

MAX_ITER = 100
N_GRASSHOPPERS = 30
N_RUNS = 10

# GOA parameters
c_max = 1.0
c_min = 0.001
f = 0.3
l = 1.5

# ====================== DATA GENERATION ======================

facilities = np.random.rand(N_FACILITIES, 2) * 100
facility_costs = np.random.randint(10, 100, size=N_FACILITIES)
customers = np.random.rand(N_CUSTOMERS, 2) * 100

dist_matrix = distance_matrix(facilities, customers)

# Normalization
dist_matrix = dist_matrix / np.median(dist_matrix)
facility_costs = facility_costs / np.max(facility_costs)

# ====================== FITNESS ======================

def evaluate_cost(solution):

    opened = np.where(solution == 1)[0]

    if len(opened) == 0:
        return np.inf

    opening_cost = np.sum(facility_costs[opened])

    transport_cost = np.sum(np.min(dist_matrix[opened, :], axis=0))

    return opening_cost + transport_cost


def evaluate_population(population):

    costs = np.zeros(len(population))

    for i, sol in enumerate(population):
        costs[i] = evaluate_cost(sol)

    return costs

# ====================== GREEDY ======================

def greedy_heuristic():

    current_solution = np.zeros(N_FACILITIES, dtype=int)
    unopened = set(range(N_FACILITIES))

    best_cost = np.inf

    while True:

        improved = False
        candidate_cost = best_cost
        candidate_facility = None

        for facility in unopened:

            temp = current_solution.copy()
            temp[facility] = 1

            cost = evaluate_cost(temp)

            if cost < candidate_cost:
                candidate_cost = cost
                candidate_facility = facility

        if candidate_facility is None:
            break

        current_solution[candidate_facility] = 1
        unopened.remove(candidate_facility)
        best_cost = candidate_cost
        improved = True

        if not improved:
            break

    return current_solution, best_cost

# ====================== BINARY TRANSFER ======================

def binary_conversion(position):

    position = np.clip(position, -10, 10)

    prob = 1 / (1 + np.exp(-10 * (position - 0.5)))

    binary = (np.random.rand(*prob.shape) < prob).astype(int)

    # prevent all-zero solution
    if np.sum(binary) == 0:
        binary[np.random.randint(N_FACILITIES)] = 1

    return binary

# ====================== GOA ======================

def GOA_UFLP():

    population = (np.random.rand(N_GRASSHOPPERS, N_FACILITIES) < 0.2).astype(int)

    best_solution = None
    best_cost = np.inf
    history = []

    for t in range(MAX_ITER):

        c = c_max - t * (c_max - c_min) / MAX_ITER

        costs = evaluate_population(population)

        best_idx = np.argmin(costs)

        if costs[best_idx] < best_cost:
            best_cost = costs[best_idx]
            best_solution = population[best_idx].copy()

        history.append(best_cost)

        new_population = population.copy()

        for i in range(N_GRASSHOPPERS):

            S = np.zeros(N_FACILITIES)

            for j in range(N_GRASSHOPPERS):

                if i == j:
                    continue

                r = np.linalg.norm(population[i] - population[j])

                s_r = f * np.exp(-r / l) - np.exp(-r)

                S += c * s_r * (population[j] - population[i]) / (r + 1e-10)

            new_pos = population[i] + c * S + (best_solution - population[i])

            new_population[i] = binary_conversion(new_pos)

        population = new_population

    return best_solution, best_cost, history

# ====================== VISUALIZATION ======================

def plot_convergence(history):

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("GOA Convergence")
    plt.grid(True)
    plt.show()


def plot_facilities(best_solution):

    plt.figure(figsize=(8, 8))

    plt.scatter(customers[:, 0], customers[:, 1], alpha=0.3, label="Customers")
    plt.scatter(facilities[:, 0], facilities[:, 1], marker='s', label="Facilities")

    opened = np.where(best_solution == 1)[0]

    plt.scatter(
        facilities[opened, 0],
        facilities[opened, 1],
        marker='*',
        s=250,
        label="Opened Facilities"
    )

    plt.legend()
    plt.title("Facility Locations")
    plt.show()

# ====================== MAIN ======================

if __name__ == "__main__":

    start_time = time.time()

    best_sol, best_cost, history = GOA_UFLP()

    runtime = time.time() - start_time

    greedy_sol, greedy_cost = greedy_heuristic()

    trial_costs = []

    for _ in range(N_RUNS):
        _, cost, _ = GOA_UFLP()
        trial_costs.append(cost)

    trial_costs = np.array(trial_costs)

    improvement = (greedy_cost - best_cost) / greedy_cost * 100

    print("\n=== GOA RESULTS ===")
    print(f"Best Cost: {best_cost:.2f}")
    print(f"Runtime: {runtime:.2f} sec")
    print(f"Opened Facilities: {np.sum(best_sol)}")

    print("\n=== GREEDY ===")
    print(f"Cost: {greedy_cost:.2f}")
    print(f"Opened Facilities: {np.sum(greedy_sol)}")

    print(f"\nImprovement: {improvement:.2f}%")

    print("\n=== STATISTICS ===")
    print(f"Mean Cost: {np.mean(trial_costs):.2f}")
    print(f"Std Dev: {np.std(trial_costs):.2f}")
    print(f"Best Run: {np.min(trial_costs):.2f}")
    print(f"Worst Run: {np.max(trial_costs):.2f}")

    print("\nOpened Facilities:")
    print(np.where(best_sol == 1)[0])

    plot_convergence(history)
    plot_facilities(best_sol)
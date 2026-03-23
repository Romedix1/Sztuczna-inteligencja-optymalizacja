import pandas as pd
import numpy as np


df = pd.read_csv('data.csv')
caloric_values = df['Caloric Value'].values
prices = df['Cena'].values
N = len(df)

hms = 30
max_price = 200
iterations = 30000

HMCR = 0.90  
PAR = 0.30   


ratios = caloric_values / prices

worst_ratios_idx = np.argsort(ratios)

def fitness(solution):
    total_price = np.sum(solution * prices)
    if total_price > max_price:
        return 0
    return np.sum(solution * caloric_values)

def repair(solution):
    total_price = np.sum(solution * prices)
    if total_price <= max_price:
        return solution
        
    for idx in worst_ratios_idx:
        if solution[idx] == 1:
            solution[idx] = 0
            total_price -= prices[idx]
            if total_price <= max_price:
                break
    return solution

def init_HM():
    HM = np.zeros((hms, N), dtype=int)
    for i in range(hms):
        sol = np.zeros(N, dtype=int)
        indices = np.random.permutation(N)
        total_price = 0
        for idx in indices:
            if total_price + prices[idx] <= max_price:
                sol[idx] = 1
                total_price += prices[idx]
        HM[i] = sol
    return HM

def generate_new_harmony(HM):
    new_sol = np.zeros(N, dtype=int)
    for i in range(N):
        if np.random.rand() < HMCR:
            random_hm_idx = np.random.randint(0, hms)
            new_sol[i] = HM[random_hm_idx, i]
            
            if np.random.rand() < PAR:
                new_sol[i] = 1 - new_sol[i] 
        else:
            new_sol[i] = np.random.randint(0, 2)
            
    return repair(new_sol)

def run_harmony_search():
    HM = init_HM()
    fitness_values = np.array([fitness(sol) for sol in HM])

    for _ in range(iterations):
        new_sol = generate_new_harmony(HM)
        new_fit = fitness(new_sol)

        worst_idx = np.argmin(fitness_values)

        if new_fit > fitness_values[worst_idx]:
            HM[worst_idx] = new_sol
            fitness_values[worst_idx] = new_fit

    best_idx = np.argmax(fitness_values)
    best_sol = HM[best_idx]

    return best_sol, fitness_values[best_idx], np.sum(best_sol * prices)

sol, cal, price = run_harmony_search()

print("Harmony Search")
print(f"Kalorie: {cal:.2f} kcal")
print(f"Koszt:   {price:.2f} PLN")
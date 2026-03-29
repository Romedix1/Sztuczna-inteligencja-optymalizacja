import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import threading
import time

def run_hs_algorithm(hms, hmcr, par, iterations, budget_limit, result_var, run_button):
    try:
        start_time = time.time()

        df = pd.read_csv('data.csv')

        df = df[df['Cena'] > 0].reset_index(drop=True)

        caloric_values = df['Caloric Value'].values
        prices = df['Cena'].values
        N = len(df)

        ratios = caloric_values / prices

        sort_asc = np.argsort(ratios)
        sort_desc = sort_asc[::-1]

        np.random.seed(42)
        HM = np.zeros((hms, N), dtype=int)
        HM_fitness = np.zeros(hms)

        for i in range(hms):
            current_cost = 0
            current_calories = 0
            for idx in np.random.permutation(N):
                if current_cost + prices[idx] <= budget_limit:
                    HM[i, idx] = 1
                    current_cost += prices[idx]
                    current_calories += caloric_values[idx]
            HM_fitness[i] = current_calories

        for iteration in range(iterations):
            rand_hmcr = np.random.rand(N)
            rand_par = np.random.rand(N)
            random_hm_indices = np.random.randint(0, hms, size=N)

            from_hm_mask = rand_hmcr < hmcr
            new_harmony = np.zeros(N, dtype=int)
            new_harmony[from_hm_mask] = HM[random_hm_indices[from_hm_mask], np.where(from_hm_mask)[0]]

            mutate_mask = from_hm_mask & (rand_par < par)
            new_harmony[mutate_mask] = 1 - new_harmony[mutate_mask]

            new_harmony[~from_hm_mask] = np.random.randint(0, 2, size=np.sum(~from_hm_mask))

            new_cost = np.sum(new_harmony * prices)

            if new_cost > budget_limit:
                for idx in sort_asc:
                    if new_harmony[idx] == 1:
                        new_harmony[idx] = 0
                        new_cost -= prices[idx]
                        if new_cost <= budget_limit:
                            break

            remaining_budget = budget_limit - new_cost
            if remaining_budget > 0:
                for idx in sort_desc:
                    if new_harmony[idx] == 0 and prices[idx] <= remaining_budget:
                        new_harmony[idx] = 1
                        remaining_budget -= prices[idx]

            new_fitness = np.sum(new_harmony * caloric_values)
            worst_index = np.argmin(HM_fitness)

            if new_fitness > HM_fitness[worst_index]:
                HM[worst_index] = new_harmony
                HM_fitness[worst_index] = new_fitness

        best_index = np.argmax(HM_fitness)
        best_harmony = HM[best_index]
        best_fitness = HM_fitness[best_index]
        best_cost = np.sum(best_harmony * prices)
        selected_count = np.sum(best_harmony)

        elapsed_time = time.time() - start_time

        result_text = (f"🌟 Znaleziono optymalny zestaw:\n\n"
                       f"Kalorie: {best_fitness:.2f} kcal\n"
                       f"Koszt: {best_cost:.2f} PLN\n"
                       f"Ilość produktów: {selected_count} z {N}\n\n"
                       f"Czas obliczeń: {elapsed_time:.2f} s")

        result_var.set(result_text)

    except FileNotFoundError:
        messagebox.showerror("Błąd", "Nie znaleziono pliku 'data.csv'.")
        result_var.set("Gotowy do uruchomienia.")
    except Exception as e:
        messagebox.showerror("Błąd uruchomienia", f"Wystąpił błąd: {str(e)}")
        result_var.set("Gotowy do uruchomienia.")
    finally:
        run_button.config(state=tk.NORMAL, text="Uruchom Optymalizację")


def start_optimization():
    try:
        hms_val = int(entry_hms.get())
        hmcr_val = float(entry_hmcr.get())
        par_val = float(entry_par.get())
        iter_val = int(entry_iterations.get())

        if not (0 <= hmcr_val <= 1 and 0 <= par_val <= 1):
            raise ValueError("HMCR i PAR muszą być w przedziale [0, 1]")

        run_btn.config(state=tk.DISABLED, text="Obliczanie...")
        result_var.set("Trwa obliczanie.")

        budget_limit = 200.0
        threading.Thread(target=run_hs_algorithm,
                         args=(hms_val, hmcr_val, par_val, iter_val, budget_limit, result_var, run_btn),
                         daemon=True).start()

    except ValueError as ve:
        messagebox.showwarning("Błąd danych wejściowych", str(ve))


root = tk.Tk()
root.title("Optymalizator kalorii")
root.geometry("380x320")
root.resizable(False, False)

result_var = tk.StringVar(value="Gotowy do uruchomienia.")

style = ttk.Style()
style.configure("TButton", font=("Arial", 10, "bold"))
style.configure("TLabel", font=("Arial", 10))

frame = ttk.Frame(root, padding="15")
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="HMS (Rozmiar Pamięci):").grid(column=0, row=0, sticky=tk.W, pady=4)
entry_hms = ttk.Entry(frame, width=15)
entry_hms.insert(0, "30")
entry_hms.grid(column=1, row=0, pady=4)

ttk.Label(frame, text="HMCR (Szansa pobrania z pamięci):").grid(column=0, row=1, sticky=tk.W, pady=4)
entry_hmcr = ttk.Entry(frame, width=15)
entry_hmcr.insert(0, "0.90")
entry_hmcr.grid(column=1, row=1, pady=4)

ttk.Label(frame, text="PAR (Szansa mutacji genów):").grid(column=0, row=2, sticky=tk.W, pady=4)
entry_par = ttk.Entry(frame, width=15)
entry_par.insert(0, "0.30")
entry_par.grid(column=1, row=2, pady=4)

ttk.Label(frame, text="Liczba Iteracji:").grid(column=0, row=3, sticky=tk.W, pady=4)
entry_iterations = ttk.Entry(frame, width=15)
entry_iterations.insert(0, "20000")
entry_iterations.grid(column=1, row=3, pady=4)

run_btn = ttk.Button(frame, text="Uruchom", command=start_optimization)
run_btn.grid(column=0, row=4, columnspan=2, pady=15, sticky=tk.W + tk.E)

result_label = ttk.Label(frame, textvariable=result_var, justify=tk.LEFT, relief=tk.SUNKEN, padding=10)
result_label.grid(column=0, row=5, columnspan=2, sticky=tk.W + tk.E)

root.mainloop()

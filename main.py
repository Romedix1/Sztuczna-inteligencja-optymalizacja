import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import threading


def run_hs_algorithm(hms, hmcr, par, iterations, budget_limit, result_var, run_button):
    try:
        df = pd.read_csv('data.csv')
        caloric_values = df['Caloric Value'].values
        prices = df['Cena'].values
        N = len(df)

        # Inicjalizacja
        np.random.seed(42)
        HM = np.zeros((hms, N), dtype=int)
        HM_fitness = np.zeros(hms)

        for i in range(hms):
            current_cost = 0
            current_calories = 0
            indices = np.random.permutation(N)
            for idx in indices:
                if current_cost + prices[idx] <= budget_limit:
                    HM[i, idx] = 1
                    current_cost += prices[idx]
                    current_calories += caloric_values[idx]
            HM_fitness[i] = current_calories

        # Pętla główna
        for iteration in range(iterations):
            new_harmony = np.zeros(N, dtype=int)

            for j in range(N):
                if np.random.rand() < hmcr:
                    random_hm_index = np.random.randint(hms)
                    new_harmony[j] = HM[random_hm_index, j]
                    if np.random.rand() < par:
                        new_harmony[j] = 1 - new_harmony[j]
                else:
                    new_harmony[j] = np.random.randint(2)

            new_cost = np.sum(new_harmony * prices)
            if new_cost > budget_limit:
                selected_indices = np.where(new_harmony == 1)[0]
                ratios = caloric_values[selected_indices] / (prices[selected_indices] + 1e-6)
                sorted_by_ratio = selected_indices[np.argsort(ratios)]

                for idx in sorted_by_ratio:
                    new_harmony[idx] = 0
                    new_cost -= prices[idx]
                    if new_cost <= budget_limit:
                        break

            remaining_budget = budget_limit - new_cost
            unselected_indices = np.where(new_harmony == 0)[0]
            ratios_unselected = caloric_values[unselected_indices] / (prices[unselected_indices] + 1e-6)
            sorted_unselected = unselected_indices[np.argsort(ratios_unselected)[::-1]]

            for idx in sorted_unselected:
                if prices[idx] <= remaining_budget:
                    new_harmony[idx] = 1
                    remaining_budget -= prices[idx]

            new_fitness = np.sum(new_harmony * caloric_values)

            worst_index = np.argmin(HM_fitness)
            if new_fitness > HM_fitness[worst_index]:
                HM[worst_index] = new_harmony
                HM_fitness[worst_index] = new_fitness

        # Wyniki
        best_index = np.argmax(HM_fitness)
        best_harmony = HM[best_index]
        best_fitness = HM_fitness[best_index]
        best_cost = np.sum(best_harmony * prices)
        selected_count = np.sum(best_harmony)

        result_text = (f"Najlepsze kalorie: {best_fitness} kcal\n"
                       f"Koszt: {best_cost:.2f} PLN\n"
                       f"Ilość produktów: {selected_count} z {N}")

        # Aktualizacja GUI z powrotem w głównym wątku
        result_var.set(result_text)

    except Exception as e:
        messagebox.showerror("Błąd uruchomienia", f"Wystąpił błąd: {str(e)}")
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
        result_var.set("Trwa obliczanie. Czekaj...")

        # Uruchomienie algorytmu w osobnym wątku, aby nie zamrozić GUI
        threading.Thread(target=run_hs_algorithm,
                         args=(hms_val, hmcr_val, par_val, iter_val, 200.0, result_var, run_btn),
                         daemon=True).start()
    except ValueError as ve:
        messagebox.showwarning("Błąd danych wejściowych", str(ve))


# Konfiguracja głównego okna
root = tk.Tk()
root.title("Harmony Search - Optymalizator Diety")
root.geometry("350x300")
root.resizable(False, False)

# Zmienne
result_var = tk.StringVar(value="Gotowy do uruchomienia.")

# Elementy GUI
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="HMS (Rozmiar Pamięci):").grid(column=0, row=0, sticky=tk.W, pady=2)
entry_hms = ttk.Entry(frame)
entry_hms.insert(0, "30")
entry_hms.grid(column=1, row=0, pady=2)

ttk.Label(frame, text="HMCR (Współczynnik HM):").grid(column=0, row=1, sticky=tk.W, pady=2)
entry_hmcr = ttk.Entry(frame)
entry_hmcr.insert(0, "0.90")
entry_hmcr.grid(column=1, row=1, pady=2)

ttk.Label(frame, text="PAR (Współczynnik Mutacji):").grid(column=0, row=2, sticky=tk.W, pady=2)
entry_par = ttk.Entry(frame)
entry_par.insert(0, "0.30")
entry_par.grid(column=1, row=2, pady=2)

ttk.Label(frame, text="Liczba Iteracji:").grid(column=0, row=3, sticky=tk.W, pady=2)
entry_iterations = ttk.Entry(frame)
entry_iterations.insert(0, "20000")
entry_iterations.grid(column=1, row=3, pady=2)

run_btn = ttk.Button(frame, text="Uruchom Optymalizację", command=start_optimization)
run_btn.grid(column=0, row=4, columnspan=2, pady=15)

# Pole wyników
result_label = ttk.Label(frame, textvariable=result_var, justify=tk.LEFT, relief=tk.SUNKEN, padding=5)
result_label.grid(column=0, row=5, columnspan=2, sticky=tk.W + tk.E)

root.mainloop()
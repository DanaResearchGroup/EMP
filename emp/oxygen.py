def compute_oxygen_balance(formula):
    # Parse the molecular formula to count the number of atoms
    n_O = formula.count('O')
    n_C = formula.count('C')
    n_N = formula.count('N')
    n_H = formula.count('H')

    # Calculate the oxygen balance
    oxygen_balance = ((n_O - (n_C + n_H/4 + n_N/3)) / (0.5 * n_C)) * 100

    return oxygen_balance

if __name__ == "__main__":
    # Get the molecular formula as user input
    molecular_formula = input("Enter the molecular formula: ")

    # Compute and display the oxygen balance
    oxygen_balance = compute_oxygen_balance(molecular_formula)
    print(f"The oxygen balance of the molecule is: {oxygen_balance:.2f}%")

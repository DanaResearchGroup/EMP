import re

def parse_reaction_equation(equation):
    # Parse the reaction equation to extract reactants and products
    reactants, products = re.split(r'\s*->\s*|\s*\+\s*', equation)
    return reactants, products

def fetch_heats_of_formation(compounds):
    # Fetch heats of formation for each compound (replace with actual data retrieval)
    heats_of_formation = [fetch_heat_of_formation(compound) for compound in compounds]
    return heats_of_formation

def fetch_heat_of_formation(compound):
    # Fetch heat of formation for a compound (replace with actual data retrieval)
    return float(input(f"Enter heat of formation for {compound}: "))

def compute_heat_of_detonation(heats_of_formation_reactants, heats_of_formation_products, sublimation_enthalpy):
    # Calculate the heat of detonation using Hess's law
    delta_H_det = sum(heats_of_formation_products) - sum(heats_of_formation_reactants)

    # Adjust for sublimation enthalpy change
    delta_H_det_adjusted = delta_H_det + sublimation_enthalpy

    return delta_H_det, delta_H_det_adjusted

def main():
    print("Heat of Detonation Calculation Tool")
    print("-----------------------------------")

    # Input reaction equation
    reaction_equation = input("Enter the balanced chemical reaction equation: ")

    # Parse the reaction equation
    reactants, products = parse_reaction_equation(reaction_equation)

    # Fetch heats of formation for reactants and products
    heats_of_formation_reactants = fetch_heats_of_formation(reactants)
    heats_of_formation_products = fetch_heats_of_formation(products)

    # Input sublimation enthalpy change
    sublimation_enthalpy = float(input("Enter the sublimation enthalpy change: "))

    # Compute the heat of detonation
    delta_H_det, delta_H_det_adjusted = compute_heat_of_detonation(
        heats_of_formation_reactants, heats_of_formation_products, sublimation_enthalpy
    )

    # Display the results
    print("\nHeat of Detonation Calculation Results:")
    print(f"Heats of Formation (Reactants): {heats_of_formation_reactants}")
    print(f"Heats of Formation (Products): {heats_of_formation_products}")
    print(f"Sublimation Enthalpy Change: {sublimation_enthalpy}")
    print(f"Heat of Detonation: {delta_H_det} kJ/mol")
    print(f"Heat of Detonation (Adjusted for Sublimation): {delta_H_det_adjusted} kJ/mol")

# Run the script
if __name__ == "__main__":
    main()

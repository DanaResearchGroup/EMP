import re

def parse_gaussian_output(output_file_path):
    with open(output_file_path, 'r') as f:
        output_text = f.read()

    # Extract relevant information related to detonation velocity
    # Replace these regular expressions with actual patterns in your Gaussian output
    energy_match = re.search(r"Total Energy\s+=\s+(-?\d+\.\d+)", output_text)
    bond_energy_match = re.search(r"Bond Energy\s+=\s+(-?\d+\.\d+)", output_text)
    reaction_enthalpy_match = re.search(r"Reaction Enthalpy\s+=\s+(-?\d+\.\d+)", output_text)

    if energy_match is None or bond_energy_match is None or reaction_enthalpy_match is None:
        raise ValueError("Unable to extract relevant information from Gaussian output")

    total_energy = float(energy_match.group(1))
    bond_energy = float(bond_energy_match.group(1))
    reaction_enthalpy = float(reaction_enthalpy_match.group(1))

    return total_energy, bond_energy, reaction_enthalpy

def compute_detonation_velocity(total_energy, bond_energy, reaction_enthalpy):
    # Perform calculations based on the extracted information
    detonation_velocity = total_energy + bond_energy - reaction_enthalpy

    return detonation_velocity

if __name__ == "__main__":
    gaussian_output_file_path = 'gaussian_output.log'

    # Parse Gaussian output and compute detonation velocity-related information
    try:
        total_energy, bond_energy, reaction_enthalpy = parse_gaussian_output(gaussian_output_file_path)
        detonation_velocity = compute_detonation_velocity(total_energy, bond_energy, reaction_enthalpy)

        print(f"Total Energy: {total_energy}")
        print(f"Bond Energy: {bond_energy}")
        print(f"Reaction Enthalpy: {reaction_enthalpy}")
        print(f"Detonation Velocity: {detonation_velocity}")
    except ValueError as e:
        print(f"Error: {e}")

import re

def parse_gaussian_output(output_file_path):
    with open(output_file_path, 'r') as f:
        output_text = f.read()

    # Extract total charge and volume from Gaussian output
    total_charge_match = re.search(r"Total Charge\s+=\s+(-?\d+)", output_text)
    volume_match = re.search(r"Volume\s+=\s+([\d.]+)\s+Bohr\^3", output_text)

    if total_charge_match is None or volume_match is None:
        raise ValueError("Unable to extract total charge or volume from Gaussian output")

    total_charge = int(total_charge_match.group(1))
    volume = float(volume_match.group(1))

    return total_charge, volume

def compute_density(total_charge, volume):
    # Density = Total Charge / Volume
    density = total_charge / volume
    return density

if __name__ == "__main__":
    gaussian_output_file_path = 'gaussian_output.log'

    # Parse Gaussian output and compute density
    try:
        total_charge, volume = parse_gaussian_output(gaussian_output_file_path)
        density = compute_density(total_charge, volume)

        print(f"Total Charge: {total_charge}")
        print(f"Volume: {volume} Bohr^3")
        print(f"Density: {density} e/Bohr^3")
    except ValueError as e:
        print(f"Error: {e}")

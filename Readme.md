# Multifractal Analysis and Simulation Toolkit

This project includes tools and scripts designed for multifractal analysis, simulations, and visualization of multifractal characteristics. The modules enable the calculation, simulation, and study of multifractal measures from various perspectives.

## Project Structure

- **`graphs.py`**: Tools for generating and visualizing graphs related to multifractal analysis.
- **`mmar_simulation.py`**: Simulation of multifractal MMAR (Multifractal Measures and Randomness) processes.
- **`multifractalcharacteristics.py`**: Calculation of multifractal characteristics such as singularity spectra.
- **`multifractal_measure_det.py`**: Analysis of deterministic multifractal measures.
- **`multifractal_measure_rand.py`**: Analysis of random multifractal measures.

## Installation

### Prerequisites

- Python 3.8 or later.
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `pandas` (optional for tabular data handling)

Install the dependencies using:

`pip install numpy matplotlib scipy pandas`

### Clone the Repository

`git clone <REPOSITORY_URL>`
`cd multifractal-analysis-toolkit`

## Usage

### Running the Modules

#### `mmar_simulation.py`

Simulate a multifractal measure:

`python mmar_simulation.py --length 1024 --hurst 0.7 --output mmar_simulation_output.txt`

#### `multifractal_measure_det.py`

Analyze a deterministic multifractal measure:

`python multifractal_measure_det.py --input data.txt --resolution 512 --output measure_analysis.png`

#### `multifractal_measure_rand.py`

Generate and analyze a random multifractal measure:

`python multifractal_measure_rand.py --length 1024 --seed 42 --output random_measure_results.txt`

#### `multifractalcharacteristics.py`

Compute multifractal characteristics:

`python multifractalcharacteristics.py --input data.txt --type "dimension_spectrum" --output spectrum_plot.png`

#### `graphs.py`

Generate graphs based on multifractal data:

`python graphs.py --input results.txt --type "scatter" --output graph.png`

### Help and Parameters

All scripts include a help option to check the available parameters:

`python <script>.py --help`

For example:

`python mmar_simulation.py --help`

## Workflow Example

1. Generate an MMAR simulation:
   `python mmar_simulation.py --length 2048 --hurst 0.5 --output simulation_data.txt`

2. Compute multifractal characteristics:
   `python multifractalcharacteristics.py --input simulation_data.txt --type "dimension_spectrum" --output spectrum.png`

3. Visualize the results:
   `python graphs.py --input simulation_data.txt --type "line" --output visualization.png`

## Contributions

Contributions are welcome. Please open an issue or submit a pull request in the repository.

# On Bits and Bandits: Quantifying the Regret-Information Trade-off
This repo implements the experiments for the paper [On Bits and Bandits: Quantifying the Regret-Information Trade-off](Insert link here).

## Requirements
Before installing the required packages, make sure that you have Python 3.10 or later installed.
Afterward, make sure you have cuda toolkit installed. To install cuda toolkit, run:
```bash
conda install nvidia::cuda-toolkit
```
To install the required packages, run:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running the Experiments
The paper has three main experiments:
1. The first experiment compares thompson sampling with three different priors. To run it, execute:
```bash
python "Stochastic MAB/different_prior.py"
```
2. The second experiment compares three different bandit algorithms. To run it, execute:
```bash
python "Stochastic MAB/different_algs.py"
```
3. The third experiment evaluates the performance of LLMs. 
Before running the experiment, you need to login to the Hugging Face model hub. To do so, run:
```bash
huggingface-cli login
```
and follow the instructions. To run the experiments, execute:
```bash
cd "MCQA task"
python evaluate_llm.py --seed [] [--large]
```
This file evaluates the LLM on the environment described in the paper. 
The --large flag specifies whether to use the large LLM. The seed flag specifies which seed to use.
The seed flag specifies which seed to use. In the paper we used the following seeds: 123,
123456, 123456789, 667,7890, 54678, 75867, 1738, 609854, 16785. The large flag specifies whether to
evaluate using the large LLM. For each seed you use run using the following LLM. You can get a graph for a specific seed by running:
```bash
cd "Contextual MAB"
python print_results.py --seed []
```
For printing the means over multiple seeds, simply run
```bash
cd "Contextual MAB"
python print_results.py --seed [<List of seeds>] --print
```

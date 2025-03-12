# Mutant Generation Tool Overview 

This project is a mutant generation tool developed in Python using PyTorch. It facilitates the generation of mutants of textual data based on predefined replacement rules. Mutants can be generated either through simple word replacement or through intersectional replacements involving multiple word lists. 

## Features 

- **Word Replacement Mutations:** Generate mutants by replacing specific words with replacements provided in a predefined dictionary. 
- **Intersectional Mutations:** Generate mutants by replacing words from two different lists while ensuring that replacements from one list do not coincide with replacements from the other list. 
- **Invariant Check:** Mutants are checked for structural similarity to the original text to ensure meaningful mutations. 

## Dependencies 

- Python 3.x 
- PyTorch 

## Installation 

Clone the repository to your local machine and navigate to the "Mutant_tool_usage" directory: 

```bash 
#git clone ...
cd Mutant_tool_usage 
``` 

Install the required dependencies, preferably in a fresh environment: 

```bash 
pip install -r requirements.txt 
``` 

## Usage 

In "mutation_llms.py," there is a simple-to-use method called `mutation` that creates mutants and performs an invariant check. To use it, simply import "mutation_llms" and fix possible issues to import "methods.py" and "sememe_coref.py", then you will be able to use it with your Python code. 

The `mutation` method takes 3 arguments: 

### Input 

```python 
def mutation(dataset_text, dict_path, inter_dict_path="") 

     dataset_text: An array of strings containing all the texts to mutate. 
     dict_path: A string representing the path to the file containing pairs of words. Refer to the files in the "data" folder at the root for examples. 
     inter_dict_path: A string representing the path to a file containing pairs of words as before. It is used to perform intersectional replacement. 
``` 

### Output 

The method will output an array containing 2 elements: 

1. The first element is a dictionary with: 
     - The time taken to generate. 
     - The number of occurrences replaced. 
     - The header of the second element. 
     - The discarded test cases, specifically the test cases that were passed because the string used was not formatted as a normal sentence. 
2. The second element is an array of multiple elements: 
     - The first word replaced in the sentence. 
     - The word it was replaced by. 
     - The second word replaced in the sentence if intersectional. 
     - The word it was replaced by if intersectional. 
     - The mutant generated. 
     - The index of the original text in the provided array in input. 
     - The original text. 
     - The similarity result after the invariant check, True if passed else False. The invariant check is very strict and may discard well-generated mutants, but the mutants that pass it are likely to be well generated. 

### Example 

```python 
# Generate mutants for atomic gender mutation 
ex1 = mutation(["The man.", "The black man."], "../data/gender/male_female.csv") 

# Generate mutants for intersectional race and gender mutation 
ex2 = mutation(["The man.", "The black man."], "../data/race/minority_majority.csv", "../data/gender/male_female.csv") 
``` 
Feel free to adjust the parameters and paths according to your specific use case. 

**Note**: The generation may take some time with many pair of words or texts.

**Note**: The project is still ongoing, therefore multiple dependencies in the requirement file may not be necessary anymore.


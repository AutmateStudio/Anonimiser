from Setup import ner_pipe
from examples import example_1, example_2, example_3, example_4

def executer_ner(text):
    ner_results = ner_pipe(text)
    return ner_results

print(executer_ner(example_1))

print("_" * 10)

print(executer_ner(example_2))

print("_" * 10)

print(executer_ner(example_3))

print("_" * 10)

print(executer_ner(example_4))

print("_" * 10)
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("rajpurkar/squad")
df = dataset['validation'].to_pandas()
print(df.info())
print(df.head())
print(len(df))

df['output'] = df['answers'].map(lambda x: x['text'][0])
df = df.drop(columns=['answers'])
print(df.head())

df.sample(1024).to_parquet("data/squad_for_llms.parquet")

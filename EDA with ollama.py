import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\titanic_ dataset_final.csv'
df = pd.read_csv(url)
df

df.describe()
sns.countplot(x = 'Survived',data = df)
plt.title('Survival count')
plt.show

import ollama
def generate_insight(df_summary):
    prompt = f'Analyze the dataset summary and provide insight:\n\n{df_summary}'
    response = ollama.chat(model = 'gemma3:270m',messages=[{'role':'user',"content":prompt}])
    
summary = df.describe().to_string()
insight = generate_insight(summary)
print("\n AI-Genrated Insight:\n", insight)

#frontend
import gradio as gr
def eda_analysis(file):
    df = pd.read_csv(file.name)
    summary = df.describe().to_string()
    insight = generate_insight(summary)
    return insight

demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(label="Upload CSV file"),
    outputs=gr.Textbox(label="AI Generated Insight"),
    title="AI powered EDA with Vaibhav Shinde"
)


demo.launch(share = True)








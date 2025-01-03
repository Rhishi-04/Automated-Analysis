# Requires Python >= 3.11
# Dependencies: httpx, pandas, seaborn, matplotlib, tenacity

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import json
from tenacity import retry, stop_after_attempt, wait_exponential

def load_data(filename):
    """Load dataset from a CSV file."""
    try:
        return pd.read_csv(filename, encoding='ISO-8859-1')
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def analyze_data(df):
    """Generate a summary analysis of the dataset."""
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'describe': df.describe(include='all').to_dict()
    }
    return summary

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_llm(prompt):
    """Call the LLM API with a given prompt."""
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = httpx.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()

def plot_missing_values(df, directory):
    """Generate and save a heatmap for missing values in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig(os.path.join(directory, 'missing_values.png'))
    plt.close()

def plot_correlation_matrix(df, directory):
    """Generate and save a correlation matrix heatmap."""
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(directory, 'correlation_matrix.png'))
    plt.close()

def write_readme(analysis_summary, images, directory):
    """Generate a README.md file summarizing the analysis and linking visualizations."""
    readme_path = os.path.join(directory, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Analysis Report\n\n")
        f.write("## Data Summary\n")
        f.write(f"Shape: {analysis_summary['shape']}\n")
        f.write(f"Columns: {', '.join(analysis_summary['columns'])}\n")
        f.write("Missing Values:\n")
        for col, count in analysis_summary['missing_values'].items():
            f.write(f"  - {col}: {count} missing values\n")
        f.write("\n")
        f.write("Describe Statistics:\n")
        f.write(json.dumps(analysis_summary['describe'], indent=2))
        
        f.write("\n\n## LLM Insights\n")
        f.write(analysis_summary['llm_insights'])
        
        f.write("\n\n## Visualizations\n")
        for image in images:
            f.write(f"![{image}]({image})\n")


def main(filename, output_directory):
    """Main function to orchestrate the analysis."""
    # Create the directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    df = load_data(filename)
    analysis_summary = analyze_data(df)
    
    # Call LLM for additional insights
    llm_prompt = f"Analyze the following data summary: {json.dumps(analysis_summary)}"
    llm_response = call_llm(llm_prompt)
    
    # Append LLM insights to the summary
    analysis_summary['llm_insights'] = llm_response.get('choices', [{}])[0].get('text', 'No insights provided.')
    
    # Generate visualizations
    plot_missing_values(df, output_directory)
    plot_correlation_matrix(df, output_directory)
    
    # Write output to README.md
    write_readme(analysis_summary, ['missing_values.png', 'correlation_matrix.png'], output_directory)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python autolysis.py <filename.csv> <output_directory>")
    else:
        main(sys.argv[1], sys.argv[2])
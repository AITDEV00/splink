import pandas as pd
try:
    df = pd.read_excel("/home/jyao/ait-projects/splink/get-started/Entity List.xlsx")
    print(df.head().to_markdown())
    print("\nColumns:", df.columns.tolist())
except Exception as e:
    print(e)

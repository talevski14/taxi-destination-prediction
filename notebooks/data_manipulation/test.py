import pandas as pd
df = {
    "CALL_TYPE": ["A", "B", "C", "A", "B", "C"],
    "LENGTH": [120, 250, 300, 180, 90, 400]
}
df = pd.DataFrame(df, columns=["CALL_TYPE", "LENGTH"])
df = df[df['CALL_TYPE'] != 'A']
print(df.head())
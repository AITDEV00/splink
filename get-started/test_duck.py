import duckdb
import pandas as pd

con = duckdb.connect()

df = pd.DataFrame({
    'a': [[0.1, 0.2, 0.3]],
    'b': [[0.1, 0.2, 0.3]]
})

# Register df
con.register('df', df)

try:
    print("Trying array_cosine_similarity...")
    res = con.execute("SELECT array_cosine_similarity(a, b) FROM df").fetchall()
    print("Success:", res)
except Exception as e:
    print("Failed array_cosine_similarity:", e)

try:
    print("Trying list_cosine_similarity...")
    res = con.execute("SELECT list_cosine_similarity(a, b) FROM df").fetchall()
    print("Success:", res)
except Exception as e:
    print("Failed list_cosine_similarity:", e)

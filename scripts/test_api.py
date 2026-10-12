from be_datahive import be_datahive

api=be_datahive()

df=api.get_efficiency(
    max_rows=100,
    limit=100,
    offset=0,
    base_editor="ABE8e"
)

print(df.shape)
print(df.columns)
print(df.base_editor.unique())
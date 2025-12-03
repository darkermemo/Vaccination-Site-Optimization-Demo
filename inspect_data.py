import pandas as pd

try:
    vacc = pd.read_excel('Vaccination_Centers_Table.xlsx')
    vill = pd.read_excel('Village_Centers_Table.xlsx')
    
    print("=== Vaccination Centers Table (First 5 rows) ===")
    print(vacc.head().to_markdown(index=False, numalign="left", stralign="left"))
    print("\n=== Village Centers Table (First 5 rows) ===")
    print(vill.head().to_markdown(index=False, numalign="left", stralign="left"))
except Exception as e:
    print(e)

import pdfplumber
import pandas as pd

pdf_path = r"C:\Users\ggpsk\OneDrive\Desktop\SIH\TERRANEX\landslide_report.pdf"
all_tables = []

with pdfplumber.open(pdf_path) as pdf:
    print("Total pages:", len(pdf.pages))

    for page_number, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()

        for table in tables:
            if table:
                all_tables.extend(table)

        print(f"Processed page {page_number}")

df = pd.DataFrame(all_tables)

df.to_csv("landslide_inventory_raw.csv", index=False)

print("\nData extraction completed!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print(df.head())
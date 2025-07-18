from langchain_community.document_loaders.csv_loader import CSVLoader


file_path = "../resource/Sheet1.csv"

loader = CSVLoader(file_path)
data = loader.load()

for record in data[:2]:
    print(record)
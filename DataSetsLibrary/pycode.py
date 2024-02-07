from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# print(drug_sample[0]) range(1000) = [0, 1, 2, ..., 999]

drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", 
                                          new_column_name="patient_id")
print(drug_dataset)
# print(drug_sample[:2])

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("patient_id"))

#new feature added to the main program
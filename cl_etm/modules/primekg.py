import os 
import pandas as pd

class PrimeKG():
    def __init__(
            self, 
            data_dir : str,
            process_node_lst : set[str] = {"drug", "disease"},
            process_edge_lst : set[str] = {},
            ):
        
        try:
            from tdc.resource import PrimeKG

            primekg = PrimeKG(path = data_dir)
            df = primekg.df
            
        except ModuleNotFoundError:
            csv_file = f"{data_dir}/kg.csv"
            
            if not os.path.exists(csv_file):
                os.system(f"wget -O {csv_file} https://dataverse.harvard.edu/api/access/datafile/6180620")

            df = pd.read_csv(csv_file, low_memory=False)

        if process_node_lst:
            df = df[df['x_type'].isin(list(process_node_lst)) & df['y_type'].isin(list(process_node_lst))]
        
        if process_edge_lst:
            df = df[df['relation'].isin(list(process_edge_lst))]

        self.df = df 

    def get_drug_disease_relation(self):
        # Filter the DataFrame to include only rows where x_type is 'drug'
        drug_df = self.df[self.df['x_type'] == 'drug']

        # Create a dictionary to store the special relations
        special_relations = {}

        # Iterate through the filtered DataFrame and populate the dictionary
        for _, row in drug_df.iterrows():
            drug_name = row['x_name']  # Assuming you want to use x_id as the drug name
            disease_name = row['y_name']
            
            if drug_name not in special_relations:
                special_relations[drug_name] = []
            
            if disease_name not in special_relations[drug_name]:
                special_relations[drug_name].append(disease_name)

        return special_relations

if __name__ == "__main__":
    
    primekg = PrimeKG("./data")
    print(primekg.get_drug_disease_relation())
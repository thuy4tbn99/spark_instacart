import pandas as pd
import numpy as np

rules_path = '../data/assoRules_baskets3M_50_70%/assoRules_baskets3M_50_70%.csv'
remove_popupar_path = './data_post/rules3M_50_70%_remove_popular.csv'
remove_dup_path = './data_post/rules3M_50_70%_remove_duplicate.csv'

# spark csv not have header
header_list = ["antecedent", "consequent", "confidence", 'lift']
assoRules = pd.read_csv(rules_path, names=header_list)
print(assoRules.head())

# list common items
import pickle
a_file = open("../model/most_common_dict.pkl", "rb")
common_dict = pickle.load(a_file)

five_most_common = []
for i, item in enumerate(common_dict):
    if i <5:
        five_most_common.append(item)
print(five_most_common)

# remove popular
df_copy = assoRules.copy()

def remove_popular(df, common_items):
    print("This is remove_popular")
    arr_ante = df['antecedent']
    
    # this items get from common_items
    arr_ante_rm = []
    count_rm = 0
    for x in arr_ante:
        y= str(x.replace(", 'Bag of Organic Bananas'", '')\
                .replace(", 'Banana'", '')\
                .replace(", 'Organic Strawberries'", '')\
                .replace(", 'Organic Hass Avocado'", '')\
                .replace(", 'Organic Avocado'", '')\
                .replace("'Bag of Organic Bananas', ", '')\
                .replace("'Banana', ", '')\
                .replace("'Organic Strawberries', ", '')\
                .replace("'Organic Hass Avocado', ", '')\
                .replace("'Organic Avocado', ", '')\
               )
        if len(y) < len(x): 
            count_rm +=1
        arr_ante_rm.append(y)
        
    print('Number remove: ', count_rm)
    df['antecedent'] = arr_ante_rm
    return 
remove_popular(df_copy, five_most_common)
df_copy.to_csv(remove_popupar_path, index=False)


# remove duplicate
df_copy2 = df_copy.copy()

# convert type list 
from ast import literal_eval
df_copy2['antecedent'] = df_copy2['antecedent'].map(literal_eval)
df_copy2['consequent'] = df_copy2['consequent'].map(literal_eval)

# sort items in rows
from more_itertools import unique_everseen
df_copy2['antecedent'] = [list(unique_everseen(arr)) for arr in df_copy2['antecedent']]
df_copy2['consequent'] = [list(unique_everseen(arr)) for arr in df_copy2['consequent']]

# change type to str -> quickly
df_copy2['antecedent'] = df_copy2['antecedent'].astype('str')
df_copy2['consequent'] = df_copy2['consequent'].astype('str')

print('Number of rules: ', df_copy2.antecedent.count())
def remove_duplicate(df):
    df.drop_duplicates(subset=['antecedent', 'consequent'], keep='first')
    return

remove_duplicate(df_copy2)
print('Number after remove duplicate: ', df_copy2.antecedent.count())
df_copy.to_csv(remove_dup_path, index=False) # save




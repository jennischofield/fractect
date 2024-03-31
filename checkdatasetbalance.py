import glob
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
single_fractures = []
multi_fractures = []
unbroken = []
single_fracture_files = []
multi_fracture_files = []
single_fracture_ap = []
single_fracture_oblique = []
single_fracture_lat = []
single_fracture_left = []
single_fracture_right = []
multi_fracture_ap = []
multi_fracture_oblique = []
multi_fracture_lat = []
multi_fracture_left = []
multi_fracture_right = []
unbroken_ap = []
unbroken_oblique = []
unbroken_lat = []
unbroken_left = []
unbroken_right = []
list_23 = 0
list_22 = 0
list_77 = 0
list_72 = 0
other = 0
single_counter = Counter(single_fractures)
multi_counter = Counter(multi_fractures)
unbroken_counter = Counter(unbroken)
path = '/Users/jenni/Desktop/Diss_Work/folder_structure/supervisely/wrist/ann'
def gather_lists(path):
    for filename in glob.glob(os.path.join(path, '*.json')): #only process .JSON files in folder.      
        with open(filename, encoding='utf-8', mode='r') as currentFile:
            data=currentFile.read().replace('\n', '')
            keyword = json.loads(data)["tags"][0]
            tags = json.loads(data)["tags"]
            if type(keyword) is dict and keyword.get('name') == 'ao_classification':
                if ';' in keyword.get('value') :
                    multi_fractures.append(keyword.get('value'))
                    multi_fracture_files.append(filename)
                    if 'side_right' in tags:
                        multi_fracture_right.append(filename)
                    elif 'side_left' in tags:
                        multi_fracture_left.append(filename)
                    else:
                        print(f"Side Label Missing {filename} has tag {tags}")
                    if 'projection_ap' in tags:
                        multi_fracture_ap.append(filename)
                    elif 'projection_lat' in tags:
                        multi_fracture_lat.append(filename)
                    elif 'projection_oblique' in tags:
                        multi_fracture_oblique.append(filename)
                    else:
                        print(f"Projection Label Missing {filename} has tag {tags}")
                else:
                    single_fractures.append(keyword.get('value'))
                    single_fracture_files.append(filename)
                    if 'side_right' in tags:
                        single_fracture_right.append(filename)
                    elif 'side_left' in tags:
                        single_fracture_left.append(filename)
                    else:
                        print(f"Side Label Missing {filename} has tag {tags}")
                    if 'projection_ap' in tags:
                        single_fracture_ap.append(filename)
                    elif 'projection_lat' in tags:
                        single_fracture_lat.append(filename)
                    elif 'projection_oblique' in tags:
                        single_fracture_oblique.append(filename)
                    else:
                        print(f"Projection Label Missing {filename} has tag {tags}")
            else:
                unbroken.append(filename)
                if 'side_right' in tags:
                    unbroken_right.append(filename)
                elif 'side_left' in tags:
                    unbroken_left.append(filename)
                else:
                    print(f"Side Label Missing {filename} has tag {tags}")
                if 'projection_ap' in tags:
                    unbroken_ap.append(filename)
                elif 'projection_lat' in tags:
                    unbroken_lat.append(filename)
                elif 'projection_oblique' in tags:
                    unbroken_oblique.append(filename)
                else:
                    print(f"Projection Label Missing {filename} has tag {tags}")
    global single_counter, multi_counter, unbroken_counter
    single_counter = Counter(single_fractures)
    multi_counter = Counter(multi_fractures)
    unbroken_counter = Counter(unbroken)
def total_fracture_types_single():
    global list_22, list_23, list_77, list_72, other
    for key, value in single_counter.most_common():
        if '23'  in key:
            list_23 += value
        elif '22'  in key:
            list_22 += value
        elif '77'  in key:
            list_77 += value
        elif '72'  in key:
            list_72 += value
        else:
            other += value
    #print(list_22[0])
def show_balances():
    plt.pie([len(single_fracture_ap),len(single_fracture_lat),len(single_fracture_oblique)], labels = ['Single Fracture Anteroposterior', 'Single Fracture Lateral','Single Fracture Oblique'])
    plt.savefig("data_balance_figures/projection_single_fracture_split")
    plt.clf()
    plt.pie([len(multi_fracture_ap),len(multi_fracture_lat),len(multi_fracture_oblique)], labels = ['Multi Fracture Anteroposterior', 'Multi Fracture Lateral','Multi Fracture Oblique'])
    plt.savefig("data_balance_figures/projection_multi_fracture_split")
    plt.clf()
    plt.pie([len(unbroken_ap),len(unbroken_lat),len(unbroken_oblique)], labels = ['Unbroken Anteroposterior', 'Unbroken Lateral','Unbroken Oblique'])
    plt.savefig("data_balance_figures/projection_unbroken_split")
    plt.clf()
    plt.pie([len(single_fracture_left),len(single_fracture_right)], labels = ['Single Fracture Left', 'Single Fracture Right'])
    plt.savefig("data_balance_figures/side_single_fracture_split")
    plt.clf()
    plt.pie([len(multi_fracture_left),len(multi_fracture_right)], labels = ['Multi Fracture Left', 'Multi Fracture Right'])
    plt.savefig("data_balance_figures/side_multi_fracture_split")
    plt.clf()
    plt.pie([len(unbroken_left),len(unbroken_right)], labels = ['Unbroken Left', 'Unbroken Right'])
    plt.savefig("data_balance_figures/side_unbroken_split")
    plt.clf()
    final_list_single = zip([float(v) for v in single_counter.values()],[k for k in single_counter])
    df = pd.DataFrame(final_list_single, columns=['count','fracture_type'])
    df_draw = df.copy()
    df_draw.loc[df_draw['count'] < 50, 'fracture_type'] = 'other'
    df_draw = df_draw.groupby('fracture_type')['count'].sum().reset_index()
    plt.pie(df_draw['count'], labels=df_draw['fracture_type'], autopct='%1.1f%%')
    plt.savefig("data_balance_figures/fracture_type_single_split")
    plt.clf()
    final_list_multi = zip([float(v) for v in multi_counter.values()],[k for k in multi_counter])
    df = pd.DataFrame(final_list_multi, columns=['count','fracture_type'])
    df_draw = df.copy()
    df_draw.loc[df_draw['count'] < 75, 'fracture_type'] = 'other'
    df_draw = df_draw.groupby('fracture_type')['count'].sum().reset_index()
    plt.pie(df_draw['count'], labels=df_draw['fracture_type'], autopct='%1.1f%%')
    plt.savefig("data_balance_figures/fracture_type_multi_split")
    plt.clf()
    plt.pie([len(single_fractures),len(multi_fractures), len(unbroken)], labels = ['single break','multi-break','unbroken'], autopct='%1.1f%%')
    plt.savefig("data_balance_figures/single_multi_unbroken_split")
    plt.clf()
    plt.pie([list_23,list_22,list_77,list_72, other], labels = ['23 break','22 break','77 break', '72 break', 'other'], autopct='%1.1f%%')
    plt.savefig("data_balance_figures/single_fracture_general_type_split")
    plt.clf()
    plt.pie([len(single_fractures),len(unbroken)], labels = ['Single Fracture', 'Unbroken'], autopct ='%1.1f%%')
    plt.savefig("data_balance_figures/single_to_unbroken_split")
    plt.clf()

def show_fracture_totals():
    print("Single Fracture:\n")
    single_counter.most_common()
    for (value, count) in single_counter.most_common():
        print(value,count)
    print("Multi Fracture:\n")
    multi_counter.most_common()
    for (value, count) in multi_counter.most_common():
        print(value,count)
def main():
    gather_lists(path)
    total_fracture_types_single()
    show_balances()
if __name__ == "__main__":
    main()
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import pickle
import joblib


# controls workflow of this executable
def main():
    df = parse_yummly()
    create_model_from_df_cuisine(df)


# opens and parses local yummly json file
def parse_yummly():
    with open('yummly.json', 'rb') as file:
        file_json = json.load(file)
        df = create_dataframe(file_json)
        file.close()
        return df


# creates a pandas dataframe from the yummly json file that contains three columns: 1. id 2. cuisine 3. ingredients
def create_dataframe(file):
    id_list = list()
    cuisine_list = list()
    ingredient_list = list()
    for recipe in file:
        id_list.append(recipe['id'])
        cuisine_list.append(recipe['cuisine'])
        ing_string = ""
        for ing in recipe['ingredients']:
            ing_string = ing_string + " " + ing
        ingredient_list.append(ing_string)
    data = {'id': id_list,
            'cuisine': cuisine_list,
            'ingredients': ingredient_list}
    df = pd.DataFrame(data=data)
    return df


# not used in final implementation
def format_cuisine(cuisine_dictionary, cuisine_list):
    ingredient_dictionary = dict()
    for cuisine in cuisine_list:
        ingredient_string = ""
        ingredient_list = cuisine_dictionary[cuisine]
        for ingredient in ingredient_list:
            ingredient_string = ingredient_string + ingredient
        ingredient_dictionary[cuisine] = ingredient_string
    return ingredient_dictionary


# not used in final implementation
def label(df):
    cuisine_list = df.cuisine.unique()
    count_label = 0
    id_list = list()
    cuisine_list2 = list()
    ingredient_list = list()
    label_list = list()
    cuisine_label_dict = dict()
    for cuisine in cuisine_list:
        cuisine_label_dict[cuisine] = count_label
        count_label = count_label + 1
    for index, row in df.iterrows():
        id_list.append(row['id'])
        cuisine_list2.append(row['cuisine'])
        ingredient_list.append(row['ingredients'])
        label_list.append(cuisine_label_dict[row['cuisine']])
    data = {'id': id_list,
            'cuisine': cuisine_list2,
            'ingredients': ingredient_list,
            'label': label_list}
    df = pd.DataFrame(data=data)
    return df


# tokenizes the ingredients and creates a SVC model for the cuisine types
def create_model_from_df_cuisine(df):
    x_train = df['ingredients']
    y_train = df['cuisine']
    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train)
    ttt = TfidfTransformer()
    x_train_ttt = ttt.fit_transform(x_train_cv)
    clf = SVC(kernel = 'linear', probability=True).fit(x_train_ttt, y_train)

    job_save1 = 'job_cv.joblib'
    joblib.dump(cv, job_save1)
    job_save = 'job_model.joblib'
    joblib.dump(clf, job_save)


if __name__ == '__main__':
    main()
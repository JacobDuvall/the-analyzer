import pickle
import pandas as pd
from train_analyzer_from_yummly import parse_yummly
import jellyfish
import sys

# provides support for all functions called within main.py


# takes argparse of ingredients and appends them all into one string that can be analyzed by learning models
def ingredients_to_string(ingredients_list):
    ingredients_string = ""
    for ingredient in ingredients_list.ingredient:
        ingredients_string = ingredients_string + ingredient + ' '
    return ingredients_string


# takes ingredients string and shows the predicted cuisine and the predicted value of the cuisine type
def predict_cuisine(ingredients):
    clf = pickle.load(open('pickle_model.pkl', 'rb'))
    cv = pickle.load(open('pickle_cv.pkl', 'rb'))
    values = clf.predict_proba(cv.transform([ingredients]))
    df = create_probability_dataframe(values, clf.classes_)
    cuisine_df = get_top_n_from_df(df, 1)
    cuisine_confidence = float(round(cuisine_df.value * 100, 2))
    recipe_confidence = recipe_finder(ingredients, 5)
    print('Cuisine: ', cuisine_df.cuisine.iat[0].capitalize(), ' (', cuisine_confidence, '%)', sep='')
    print(recipe_confidence)


# formats value list and class list into a pandas dataframe that can more easily be analyzed for top values
def create_probability_dataframe(value_list, class_list):
    v_list = list()
    c_list = list()
    data = {'value': value_list[0],
            'cuisine': class_list}
    df = pd.DataFrame(data=data)
    return df


# retrieve the top n best columns based on model score
def get_top_n_from_df(df, n):
    largest = df.nlargest(n, 'value')
    return largest


# given the input ingredients, finds the top n most similar recipes from yummly
def recipe_finder(ingredients, n):
    df = parse_yummly()
    ingredients_list = set(ingredients.split())
    match_number = 0
    id_score_dict = dict()
    for index, row in df.iterrows():
        row_i_list = set(row['ingredients'].split())
        for ingredient in ingredients_list:
            for i in row_i_list:
                if ingredient == i:
                    match_number += 1
        match_percentage = match_number / len(ingredients_list)
        match_number = 0
        id_score_dict[row['id']] = match_percentage
    final_string = "Closest " + str(n) + " recipes: "
    for key in {key: id_score_dict[key] for key in sorted(id_score_dict, key=id_score_dict.get, reverse=True)[:n]}:
        value = ({key: id_score_dict[key] for key in sorted(id_score_dict, key=id_score_dict.get, reverse=True)[:n]}[key])
        final_string = final_string + str(key) + ' ' + '(' + str(round(value * 100, 2)) + '%) '
    return final_string



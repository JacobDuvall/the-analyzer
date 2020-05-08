import argparse
import project_3
from sklearn.svm import _classes


# takes arguments from argparse coming from --ingredient to process ingredients into a list
# that can be analyzed by learning models to predict cuisine type and N closest recipes
def main(arguments):
    ingredients = project_3.ingredients_to_string(arguments)
    project_3.predict_cuisine(ingredients)


# runs via: "py main.py --ingredient granola --ingredient rice"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # all possible arguments defined with help
    parser.add_argument("--ingredient", action='append', type=str, required=True,
                        help="ingredient for cooking!")

    args = parser.parse_args()
    if args.ingredient:
        main(args)
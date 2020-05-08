# cs5293sp20-project3
Jacob Duvall

# CHECK THIS OUT!
https://github.com/JacobDuvall/cs5293sp20-project3/blob/master/Project_3_Jupyter_Notebook_PDF.pdf

## How I turned my text into features -- and why!
In order to feature-ify my text, I processed it, set it into a dataframe for ease of use, and then applied vectorization techniques upon the text of the ingredients. To start, I downloaded the Yummly dataset in its json form from https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json. I then loaded the json file from local storage and parsed it into a dataframe. This dataframe consisted of the id, cuisine type, and ingredients list in the form of a string. With the dataframe setup, I then moved into feature-ifying the information. To do this I used Count Vectorizer to fit transform my ingredients data and then I used TfidfTransformer to fit_transform the count vectorizer data. This vectorized data was then in the right format to where I could then combine it with my cuisine data and id data to develop some learning models. 

## What classifiers/clustering methods did I choose -- and why!
Ultimately I used a Support Vector Classifer. I found this model to be quite accurate and offered a ton of features to help me accomplish some of the project goals. But the list of models I tried was more exhaustive. The major classifiers I tried were Logistic Regression, Naive Bayes, Random Forest, KMeans, Nearest Neighbors, Linear SVC, and SVC. My ideal choice was actually not my final choice here. I found Linear SVC to be slighly more accurate and scalable than SVC. It was soooo much faster at training! But the problem was that it lacked some of the features I wanted to utilize in the project that SVC had. The major feature I'm talking about and used was the probability = True parameter. Training my model with this allowed me to extract probabilities on predictions to show percentages on predictions. So for instance, if one were to run the program via "py main.py --ingredients duck" and the output was "Cuisine: French (86.12%).." -- then the model was seeing an 86.12% liklihood of the ingredients being for French cuisine. This feature kept me from needing to build my own and it was worth the downside associated with training time cost. 

## What N did I choose -- and why!
I chose 5 for my N. Seeing the closest 5 recipes is really interesting! This number can easily be changed though in my configuration due to my implementation, though. My implementation for recipe matching is easy to explain. I take all of the input ingredients and split the ingredients into a list. I then loop through all of the recipes from yummly and split all of their ingredients into a list. I then remove duplicates from both of these lists and I look for matches. If all the ingredients from my user's input are contained within the set of recipe ingredients, it is a 100% match. This match % scales to the match given. I then create a dictionary with all the recipe ID's and their match percentages and I sort the dictionary. The top N results are shown via the output. So although I chose 5 to display here, any N can be displayed. 

## Describe functions/code!
My code has 2 executables and 3 files. The 2 executables are main and train_analyzer_from_yummly. Main consists of main.py and project_3.py -- it's used to actually process ingredients provided on input and utilize trained models in order to do so. Train_analyzer_from_yummly actually handles are the work in training the models ahead of time to be utilized by the main program. 
Here's the major functionality of each executable:
1. Main.py:
- runs via: "py main.py --ingredient granola --ingredient rice"

main()
- takes arguments from argparse coming from --ingredient to process ingredients into a list that can be analyzed by learning models to predict cuisine type and N closest recipes 

Project_3.py
- provides support for all functions called within main.py

ingredients_to_string()
- takes argparse of ingredients and appends them all into one string that can be analyzed by learning models

predict_cuisine()
- takes ingredients string and shows the predicted cuisine and the predicted value of the cuisine type

create_probability_dataframe()
- formats value list and class list into a pandas dataframe that can more easily be analyzed for top values

get_top_n_from_df()
- retrieve the top n best columns based on model score

recipe_finder()
- given the input ingredients, finds the top n most similar recipes from yummly 

2. Train_analyzer_from_yummly.py:
- trains the models that I use in main for quick response and predictions

main()
- controls workflow of this executable

parse_yummly()
- opens and parses local yummly json file

create_dataframe()
- creates a pandas dataframe from the yummly json file that contains three columns: 1. id 2. cuisine 3. ingredients

create_model_from_df_cuisine()
- tokenizes the ingredients and creates a SVC model for the cuisine types


# Describe Tests!
test_model()
-tests the existence of the model for cuisine from pickle file

test_cv()
-tests the existence of the cv for cuisine from pickle file

test_user_input()
-tests that the user input for argparse is functional

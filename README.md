# cs5293sp20-project3
10 pts: How did you turn your text into features and why?
10 pts: What classifiers/clustering methods did you choose and why?
10 pts: What N did you choose and why?
5 pts: Describe functions/code
5 pts: Describe tests

## How I turned my text into features -- and why!
In order to feature-ify my text, I processed it, set it into a dataframe for ease of use, and then applied vectorization techniques upon the text of the ingredients. To start, I downloaded the Yummly dataset in its json form from https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json. I then loaded the json file from local storage and parsed it into a dataframe. This dataframe consisted of the id, cuisine type, and ingredients list in the form of a string. With the dataframe setup, I then moved into feature-ifying the information. To do this I used Count Vectorizer to fit transform my 

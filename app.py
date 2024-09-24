from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from numpy import nan


app = Flask(__name__)

df = pd.read_csv('Final.csv')

print(df.columns)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

for column in ['Hair Color:', 'Fake Boobs:', 'Piercings:', 'Background:']:
    df[column] = label_encoders[column].transform(df[column])

features = ['Age:', 'Hair Color:', 'Fake Boobs:', 'Piercings:', 'Background:', 'Height', 'Weight']

unique_values = {
    'Hair Color:': df['Hair Color:'].unique(),
    'Fake Boobs:': df['Fake Boobs:'].unique(),
    'Piercings:': df['Piercings:'].unique(),
    'Background:': df['Background:'].unique()
}


for column in unique_values:
    unique_values[column] = label_encoders[column].inverse_transform(unique_values[column])

# Add this import at the beginning of your app.py
details_df = pd.read_csv('Details.csv')

# Define the columns you want to exclude
excluded_columns = ['Unnamed: 0', 'Birthplace:', 'Interests and hobbie:','Link','Interests and hobbies:']  # Add more columns as needed

# Update the recommend_similar_star function to exclude specific columns
def recommend_similar_star(user_input):
    # Transform user inputs using label encoders
    user_input['Hair Color:'] = label_encoders['Hair Color:'].transform([user_input['Hair Color:']])[0]
    user_input['Fake Boobs:'] = label_encoders['Fake Boobs:'].transform([user_input['Fake Boobs:']])[0]
    user_input['Piercings:'] = label_encoders['Piercings:'].transform([user_input['Piercings:']])[0]
    user_input['Background:'] = label_encoders['Background:'].transform([user_input['Background:']])[0]

    # Create a DataFrame for the user input
    user_df = pd.DataFrame([user_input])

    # Filter the DataFrame to include only rows where Rank < 1500 (or other criteria)
    filtered_df = df[df['Rank'] < 1998]

    # Compute cosine similarity between user input and the filtered dataset
    similarity_scores = cosine_similarity(user_df[features], filtered_df[features])

    # Find the index of the most similar pornstar in the filtered DataFrame
    most_similar_idx = similarity_scores.argmax()

    # Get the name of the most similar pornstar
    recommended_name = filtered_df['Name'].iloc[most_similar_idx]

    # Retrieve all details from Details.csv based on the recommended name
    recommended_details = details_df[details_df['Name'] == recommended_name]

    if not recommended_details.empty:
        # Convert the row to a dictionary and filter out excluded columns
        details_dict = recommended_details.iloc[0].to_dict()
        
        # Remove excluded columns
        for col in excluded_columns:
            details_dict.pop(col, None)  # Remove the excluded columns
        
        # Replace NaN values with "Not Known"
        for key, value in details_dict.items():
            if pd.isna(value):  # Check if the value is NaN
                details_dict[key] = "Not Known"  # Replace NaN with "Not Known"

        return recommended_name, details_dict
    else:
        return recommended_name, None  # Return None if no details are found





# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_star = None
    recommended_link = None
    recommended_details = None 

    if request.method == 'POST':
        user_input = {
            'Age:': int(request.form['age']),
            'Hair Color:': request.form['hair_color'],
            'Fake Boobs:': request.form['fake_boobs'],
            'Piercings:': request.form['piercings'],
            'Background:': request.form['background'],
            'Height': float(request.form['height']),
            'Weight': float(request.form['weight'])
        }

        recommended_star, recommended_details = recommend_similar_star(user_input)

        

        images = pd.read_csv('images.csv')
        link_row = images[images['Name'] == recommended_star]
        if not link_row.empty:
            link = link_row['Link'].values[0]
            full_url = f"https://pornhub.com{link}"
            recommended_link = full_url
        else:
            recommended_link = "No link found for the predicted pornstar name."

    return render_template('index.html', unique_values=unique_values, 
                           recommended_star=recommended_star, 
                           recommended_link=recommended_link, 
                           recommended_details=recommended_details)



if __name__ == '__main__':
    app.run(debug=True)

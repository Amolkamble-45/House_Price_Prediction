import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Get the form data from the request
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        living_area = request.form['living_area']
        lot_area = request.form['lot_area']
        floors = request.form['floors']
        condition = request.form['condition']
        grade = request.form['grade']

        area_excluding_basement = request.form['area_excluding_basement']
        area_basement = request.form['area_basement']
        built_year = request.form['built_year']
        postal_code = request.form['postal_code']
        distance_airport = request.form['distance_airport']

        data = {
            'number of bedrooms': [bedrooms],
            'number of bathrooms': [bathrooms],
            'living area': [living_area],
            'lot area': [lot_area],
            'number of floors':[floors],
            'condition of the house':[condition],
            'grade of the house':[grade],
            'Area of the house(excluding basement)':[area_excluding_basement],
            'Area of the basement':[area_basement],
            'Built Year': [built_year],
            'Postal Code': [postal_code],
            'Distance from the airport':[distance_airport],


            # Add the remaining form data similarly
        }

        dataframe = pd.DataFrame(data)

        with open("random_forest_model.pickle", 'rb') as file:
            clf = pickle.load(file)
        #
        # # Make a prediction
        pred = clf.predict(dataframe)
        pred = pred.astype(int)
        value = pred[0]
        # Get the remaining form data similarly

        # Process the form data or perform any other operations
        # ...

        # Return a response or redirect to another page
        return render_template('result.html', predicted_price=value)

    # Render the HTML form
    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)

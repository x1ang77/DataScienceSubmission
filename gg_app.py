
import streamlit as st
import pickle
import datetime

# Load the pre-trained model
with open('xiang_model.h5', 'rb') as file:
    classifier = pickle.load(file)

def main():
    st.title('Sleep Efficiency Prediction App')
    st.write('Enter the following details to get the sleep efficiency prediction:')

    # Create input fields for user data
    age = st.slider('Age', 18, 100)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    caffeine_consumption = st.number_input('Caffeine Consumption (mg)', 0.0, 1000.0, 100.0)
    alcohol_consumption = st.slider('Alcohol Consumption (drinks)', 0.0, 10.0, 0.0)
    smoking_status = st.selectbox('Smoking Status', ['Yes', 'No'])
    exercise_frequency = st.slider('Exercise Frequency (days per week)', 0, 7, 3)
    light_sleep = st.slider('Light Sleep', 0, 100)
    rem_sleep = st.slider('Rem Sleep', 0, 100)
    deep_sleep = st.slider('Deep Sleep', 0, 100)
    awakenings = st.slider('Awakenings', 0, 7)
    sleep_duration = st.slider('Duration', 0, 24)

    bedtime_hour = st.slider('Bedtime (Hour)', 0, 23, 22)
    bedtime_minute = st.slider('Bedtime (Minute)', 0, 59, 0)

    # Convert the hour and minute to a datetime.time object
    bedtime = datetime.time(bedtime_hour, bedtime_minute)

    # Display the selected bedtime
    st.write(f'Selected Bedtime: {bedtime}')

    # Convert input data to appropriate format for prediction
    bedtime_hour = bedtime.hour + bedtime.minute / 60

    wakeup_hour = st.slider('Wakeup Time (Hour)', 0, 23, 7)
    wakeup_minute = st.slider('Wakeup Time (Minute)', 0, 59, 0)

    # Convert the hour and minute to a datetime.time object
    wakeup_time = datetime.time(wakeup_hour, wakeup_minute)
    wakeup_time_hour = wakeup_time.hour + wakeup_time.minute / 60

    gender_numeric = 0 if gender == 'Male' else 1
    smoking_status_numeric = 1 if smoking_status == 'Yes' else 0

    # Make the prediction
    prediction = classifier.predict([[age, gender_numeric, caffeine_consumption, alcohol_consumption,
                                      smoking_status_numeric, exercise_frequency, bedtime_hour, wakeup_time_hour, light_sleep, rem_sleep, deep_sleep, awakenings, sleep_duration]])[0]

#     light_sleep_percentage = 3
#     rem_sleep_percentage = 3
#     deep_sleep_percentage = 3
#     prediction += light_sleep_percentage + rem_sleep_percentage + deep_sleep_percentage

    # Ensure the predicted sleep efficiency is between 3 and 100
#     prediction = max(3, min(prediction, 100))

    # Display the predicted sleep efficiency
    st.write(f'Predicted Sleep Efficiency: {prediction:.2f}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
  
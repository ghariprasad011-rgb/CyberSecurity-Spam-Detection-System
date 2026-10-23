import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
#data = pd.read_csv("C:\Python\CyberSecurityDetector\spam.csv")
data = pd.read_csv(r"C:\Python\CyberSecurityDetector\spam_data.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

mess = data['Message']
Cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, Cat, test_size=0.2)

cv =CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#creating Model

model = MultinomialNB()
model.fit(features, cat_train)

#Test our model
features_test = cv.transform(mess_test)
#print(model.score(features_test,cat_test))
#predict Data
def predict(message):
 input_message = cv.transform([message]).toarray()
 result = model.predict(input_message)
 return result

st.header('Cyber Security Spam Detection :')
st.subheader('Team Members:')
st.markdown("""      
[<span style='color:purple; font-weight:bold'>Kanishka R</span>, 
<span style='color:Red; font-weight:bold'>Sameena V</span>, 
<span style='color:Brown; font-weight:bold'>Nithin S</span>, 
<span style='color:Blue; font-weight:bold'> Hari Prasad G</span>
]            
""", unsafe_allow_html=True)
if 'history' not in st.session_state:
    st.session_state['history'] = []


input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    result = predict(input_mess)
    color = 'red' if result == "Spam" else 'green'
    st.markdown(f"<h2 style='color:{color}; font-weight:bold'>{result}</h2>", unsafe_allow_html=True)
    
    # Store in history
    st.session_state['history'].append({'Message': input_mess, 'Prediction': result})

# Show history table
if st.session_state['history']:
    st.subheader("History of Messages")
    history_df = pd.DataFrame(st.session_state['history'])
    st.table(history_df)


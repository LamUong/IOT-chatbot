from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from string import punctuation
import pickle
import numpy as np
import h5py
import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

responses = [
{
"patterns": ["Ok, I'm interested to know more, can I see a demo?"],

"responses": [ "A: Definitely. You can try the demo. Come back if you have questions. Happy Exploring!"
             ],
"intent": ["General_Information","demo"]
},
{
"patterns": ["hi", "how are you", "is anyone there", "hello", "good day",
             "morning","i would like some help","test Hi there, can some one help me?"],
"responses": [ "Hi there, how can I help?"],
"intent": ["greeting","hello"]
},

{
"patterns": ["bye", "See you later", "Goodbye", "See you again","thank you for your help","test Good bye"],
"responses": ["See you later, thanks for using HelperBot."],
"intent": ["greeting","goodbye"]
},

{
"patterns": ["Hey bot, could you open the search bar", "Where is the search function","Show me the search bar","find me the search bar"
             "The location of search function","how can I access the search bar?","test Open the search bar",
             "test where can i find the search bar"],
"responses": ["The search bar is at the top on the left side-bar."],
"intent": ["search_function","search_bar_location"]
},

{
"patterns": ["search for South Korea?", "Where is Haga, Gothenburg?",
             "Could you search for Nordstan, Nordstadstorget on the map?", "Tell me the location of Neosho County?",
             "Where is location of United States?","Could you locate Nou, Sibiu, Romania?",
             "The location of Montreal?","test Where is the United States"],
"responses": ["Searching for location <   > right now."],
"intent": ["search_function","search_for_location"]
},
{
"patterns": ["Hey bot, Show me all the notifcations?", " Where are the notifications?",
             "Could you open all notifcations?","please turn on the notifcation tab.",
             "How can I access the notifcations bar?","where is the notification tab?","What notifications are there already?",
             "What regulations are there",
             "test Open the notification tab please"],
"responses": ["The notifcations tab is the second tab on the left side bar."],
"intent": ["notification_function","notification_tab_location"]
},
{
"patterns": ["Can you sort the notifications by time?","print the latest notifcations",
             "What are the earliest notifications?","show the most recent notifications",
             "can i see the most current notifications","test Give me the latest notification"],
"responses": ["Sorting the notifcations by time."],
"intent": ["notification_function","sort_by_time"]
},

{
"patterns": ["sort the notifications by importance?",
             "show the notifications by severity?","tell me the most urgent notifcations.",
             "How can I find the most urgent notifcations","What notifications are critical?",
             "print all the major notifications","show all the minor notifications",
             "test List me the most urgent notifications"],
"responses": ["Sorting the notifcations by importance."],
"intent": ["notification_function","sort_by_importance"]
},

{
"patterns": ["sort the notifications by devices?",
             "give the notifications by machine?","Give me the machines that are notified.",
             "Which machines give notifications?","What devices needed to be attended to?",
             "what devices are notified?","test Give me a list of devices on the notifications"],
"responses": ["Sorting the notifcations by device."],
"intent": ["notification_function","sort_by_device"]
},


{
"patterns": ["Show me all the alarms?","Give me all the alerts.","Show me all the issues","What are the errors"
             "Hey bot, Where are the alarms?","Could you open all the alarms?","find me all the alarms",
             "open the alarms tab.","How can I access the alarms ?","test turn on the alarm tab","test Please log all the errors"],
"responses": ["The alarm tab is the third tab on the left side bar."],
"intent": ["alarm_function","alarm_tab_location"]
},

{
"patterns": ["Can you sort the alarms by time?","Give me the latest alarms",
             "What are the earliest alarms?","show me the most recent alarms?",
             "Give me the alarms happened today.","test show all alarms happened yesterday?"],
"responses": ["Sorting the alarms by time."],
"intent": ["alarm_function","sort_by_time"]
},

{
"patterns": ["Can you sort the alarms by importance?","Give me all the alarms by severity?",
             "Show all most urgent alarms.","How can I find the most urgent alarms",
             "What alarms are critical?","list all the major alarms","test List all the critical alrams."],
"responses": ["Sorting the alarms by importance."],
"intent": ["alarm_function","sort_by_importance"]
},

{
"patterns": ["Could you open the Rules tab?", "Where are the Rules function?", "the location of Rule tab?", "What logic can I add? ",
             "What logic can I input? ","find me the rules tab"
             "What are the rules have been created", "What are rules","What custom rules are there","show me how to create rules"
             "test Where is the rule tab?"],
"responses": ["The rules tab is the fourth tab on the left side-bar."],
"intent": ["rules_function","rules_location"]
},

{
"patterns": ["change the map style?", "Show my map to be like google map?",
             "Change my map to satellite images on my map", "Display street view on map.","Can you open the map for me"
             "test  change the map too google style.","test Show me the map","Open the map for me"],
"responses": ["Open the map_function on the left side-bar. Then choose the map type you want."],
"intent": ["map_function","map_edit"]
},

{
"patterns": ["I want to go to my Profile?", "How do I check my personal information", "What is my age",
             "How can I change my personal settings?","test Open my personal information page."],
"responses": ["Open the Profile on the left side-bar."],
"intent": ["profile_function","profile_sarch"]
},

{
"patterns": ["I want to Log out?", "Please log me out", " sign me out","I want to sign out",
             "Close the application for me.", "test sign me out."],
"responses": ["The sign out button is on the Profile Tab."],
"intent": ["profile_function","log_out"]
}
]

model = load_model('model.hdf5')

infile = open("tokenizer.pkl", 'rb')
tk = pickle.load(infile)
print(tk)

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def stemmed_sentence(sentence):
    out=""
    for word in sentence.split(" "):
        word = strip_punctuation(word)
        word= porter_stemmer.stem(word)
        out+=word+" "
    return out

def preprocessing(sentence):
    MAX_SEQUENCE_LENGTH = 16
    # stem sentence
    processed_text = stemmed_sentence(sentence)
    # turn text to sequence
    sequence = tk.texts_to_sequences([processed_text])
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return sequence

def predict(sentence):
    processed_text = preprocessing(sentence)
    print(processed_text)
    y_proba=model.predict(processed_text)
    print(y_proba)
    #y_pred_index = np.argmax(keras.utils.np_utils.to_categorical(y_proba)[0])
    y_pred_index = np.argmax(y_proba[0])
    y_pred_class = responses[y_pred_index]['responses'][0]
    print(y_pred_class)
    return y_pred_class

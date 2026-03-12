import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

print("SIT Nagpur Student FAQ Chatbot")
print("Type 'exit' to stop\n")


# -----------------------------
# WEEK 2 – TEXT PREPROCESSING
# -----------------------------

stopwords = ["is","the","a","an","what","where","when","how","does","for"]

spelling = {
"collage":"college",
"feees":"fees",
"hostal":"hostel"
}

def preprocess(text):

    text = text.lower()                                   # Lowercasing
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()                                  # Tokenization
    words = [w for w in words if w not in stopwords]       # Stopword removal
    words = [spelling.get(w, w) for w in words]            # Spelling normalization

    return " ".join(words)


# -----------------------------
# WEEK 1 – RULE BASED FAQ
# -----------------------------

faq_rules = {

"timing":"College timing at SIT Nagpur is generally from 9 AM to 5 PM.",

"location":"SIT Nagpur is located in MIHAN, Nagpur, Maharashtra.",

"courses":"SIT Nagpur offers B.Tech programs in CSE, AI & ML, Mechanical and Electronics.",

"fees":"The B.Tech tuition fee at SIT Nagpur is approximately ₹3 lakh per year.",

"payment":"Students can pay SIT Nagpur fees through the official portal using net banking, debit card, credit card or UPI.",

"hostel":"SIT Nagpur provides hostel facilities for both boys and girls.",

"library":"SIT Nagpur has a digital library with books, journals and research resources.",

"placement":"Top recruiters include Infosys, TCS, Wipro and Capgemini.",

"scholarship":"SIT Nagpur provides merit-based scholarships and financial aid to eligible students.",

"contact":"You can contact SIT Nagpur at admissions@sitnagpur.edu.in.",

"admission":"Admission to SIT Nagpur B.Tech programs is through the SITEEE entrance exam.",

"campus":"SIT Nagpur campus includes modern labs, hostels, sports facilities and research centers.",

"canteen":"The SIT Nagpur campus has a student cafeteria and food court.",

"wifi":"WiFi is available across the SIT Nagpur campus for students.",

"timetable":"Class timetables are provided by departments through the student portal."
}


# -----------------------------
# WEEK 3 – SYNONYM MATCHING
# -----------------------------

synonyms = {

"fees":["fee","tuition"],

"payment":["pay","payment","pay fees","online payment"],

"scholarship":["financial aid","grant","funding"],

"hostel":["accommodation","dorm"],

"placement":["placements","company","companies","job"],

"courses":["course","branch","program"],

"admission":["apply","enrollment","registration"],

"timetable":["schedule","routine","lecture"]
}

def synonym_match(query):

    for key, words in synonyms.items():

        if key in query:
            return key

        for w in words:
            if w in query:
                return key

    return None


# -----------------------------
# WEEK 5 – INTENT CLASSIFICATION
# -----------------------------

training_questions = [

# admission
"how to apply for admission",
"admission process sit nagpur",
"when do admissions start",

# fees
"btech fees sit nagpur",
"tuition fee",

# payment
"how to pay fees",
"fee payment method",

# scholarship
"scholarship details",
"financial aid for students",

# hostel
"hostel facility",
"is hostel available",

# placement
"placement companies",
"which companies visit sit nagpur",

# courses
"courses offered",
"btech programs",

# timetable
"class timetable",
"lecture schedule"
]

labels = [

"admission","admission","admission",

"fees","fees",

"payment","payment",

"scholarship","scholarship",

"hostel","hostel",

"placement","placement",

"courses","courses",

"timetable","timetable"
]

count_vector = CountVectorizer()

X_train = count_vector.fit_transform(training_questions)

model = MultinomialNB()

model.fit(X_train, labels)


# -----------------------------
# WEEK 4 – TF-IDF FAQ RETRIEVAL
# -----------------------------

faq_questions = list(faq_rules.keys())

faq_answers = list(faq_rules.values())

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(faq_questions)


# -----------------------------
# CHATBOT LOOP
# -----------------------------

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("Bot: Thank you for contacting SIT Nagpur!")
        break


    # WEEK 2 preprocessing
    clean_query = preprocess(query)


    # WEEK 3 synonym matching
    intent = synonym_match(clean_query)

    if intent in faq_rules:
        print("Bot:", faq_rules[intent])
        continue


    # WEEK 5 intent classification
    q_vec = count_vector.transform([query])
    predicted_intent = model.predict(q_vec)[0]

    if predicted_intent in faq_rules:
        print("Bot:", faq_rules[predicted_intent])
        continue


    # WEEK 4 TFIDF retrieval fallback
    query_vec = tfidf.transform([clean_query])
    similarity = cosine_similarity(query_vec, X)

    index = similarity.argmax()

    print("Bot:", faq_answers[index])
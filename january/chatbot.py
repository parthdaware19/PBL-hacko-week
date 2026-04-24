import json
import os
import pickle
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
def download_nltk_data():
    for resource in ['punkt', 'stopwords', 'punkt_tab']:
        nltk.download(resource, quiet=True)

download_nltk_data()

from preprocess import preprocess, preprocess_to_string


# ===== WEEK 1: BASIC FAQ RESPONDER =====
# Rule-based chatbot with fixed FAQs, keyword matching, and greeting handling

GREETINGS = ['hi', 'hello', 'hey', 'howdy', 'hiya', 'good morning', 'good evening', 'good afternoon']
FAREWELLS = ['bye', 'goodbye', 'exit', 'quit', 'see you', 'later', 'cya']

def handle_greeting(user_input):
    lowered = user_input.lower().strip()
    for g in GREETINGS:
        if g in lowered:
            return "Hello! Welcome to the Campus Assistance System. How can I help you today?"
    return None

def handle_farewell(user_input):
    lowered = user_input.lower().strip()
    for f in FAREWELLS:
        if f in lowered:
            return True
    return False

KEYWORD_RULES = {
    'timing': "College timings are Monday to Friday, 8:00 AM to 5:00 PM. Saturday classes run 8:00 AM to 1:00 PM.",
    'library': "The library is open Monday to Friday 7:30 AM – 10:00 PM, and Saturday 9:00 AM – 6:00 PM.",
    'contact': "Contact admissions at admissions@campus.edu or +1-800-123-4567 (Mon–Fri, 9 AM – 4 PM).",
    'id card': "Student ID cards are issued during orientation. Replacement cards cost $10 at the student services office.",
    'portal': "Access all student services at portal.campus.edu using your student credentials.",
}

def rule_based_match(user_input):
    lowered = user_input.lower()
    for keyword, response in KEYWORD_RULES.items():
        if keyword in lowered:
            return response
    return None


# ===== WEEK 3: SYNONYM-AWARE FAQ BOT =====
# Maps synonyms to canonical terms so queries like "dorm" match "hostel" FAQs

SYNONYM_MAP = {
    # Fees
    'tuition': 'fees',
    'payment': 'fees',
    'cost': 'fees',
    'price': 'fees',
    'charge': 'fees',
    'pay': 'fees',
    # Hostel
    'accommodation': 'hostel',
    'dorm': 'hostel',
    'dormitory': 'hostel',
    'room': 'hostel',
    'residential': 'hostel',
    'housing': 'hostel',
    'stay': 'hostel',
    # Admissions
    'enroll': 'admission',
    'enrollment': 'admission',
    'register': 'admission',
    'registration': 'admission',
    'apply': 'admission',
    'application': 'admission',
    'join': 'admission',
    # Exams
    'test': 'exam',
    'assessment': 'exam',
    'evaluation': 'exam',
    'result': 'exam',
    'grade': 'exam',
    'score': 'exam',
    # Timetable
    'schedule': 'timetable',
    'timing': 'timetable',
    'time': 'timetable',
    'hour': 'timetable',
    'class': 'timetable',
    # Scholarships
    'bursary': 'scholarship',
    'grant': 'scholarship',
    'financial aid': 'scholarship',
    'stipend': 'scholarship',
    'merit': 'scholarship',
    'funding': 'scholarship',
}

def apply_synonyms(tokens):
    """Replace synonym tokens with canonical terms"""
    expanded = []
    for token in tokens:
        expanded.append(SYNONYM_MAP.get(token, token))
    return expanded

def preprocess_with_synonyms(text):
    """Preprocess text and expand synonyms"""
    tokens = preprocess(text)
    tokens = apply_synonyms(tokens)
    return ' '.join(tokens)


# ===== WEEK 4: FAQ RETRIEVAL USING TF-IDF =====
# Converts FAQ questions into TF-IDF vectors and uses cosine similarity for matching

class TFIDFFAQRetriever:
    def __init__(self, faqs):
        self.faqs = faqs
        self.questions = [f['question'] for f in faqs]
        self.answers = [f['answer'] for f in faqs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        # Preprocess questions with synonym expansion before fitting
        processed_questions = [preprocess_with_synonyms(q) for q in self.questions]
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)

    def get_best_answer(self, user_query, threshold=0.15):
        processed_query = preprocess_with_synonyms(user_query)
        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score >= threshold:
            return self.answers[best_idx], best_score, self.questions[best_idx]
        return None, best_score, None


# ===== WEEK 5: INTENT CLASSIFICATION =====
# Loads trained Naive Bayes / Logistic Regression model to detect user intent

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        # Prefer Logistic Regression, fall back to Naive Bayes
        if os.path.exists('intent_lr.pkl'):
            with open('intent_lr.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'Logistic Regression'
        elif os.path.exists('intent_nb.pkl'):
            with open('intent_nb.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'Naive Bayes'
        else:
            self.model = None
            self.model_type = None

    def predict(self, text):
        if self.model is None:
            return 'unknown', 0.0
        processed = preprocess_to_string(text)
        intent = self.model.predict([processed])[0]
        prob = max(self.model.predict_proba([processed])[0])
        return intent, round(float(prob), 3)

    def is_ready(self):
        return self.model is not None


# ===== MAIN CHATBOT — Integrates All 5 Weeks =====

class CampusChatbot:
    def __init__(self, faq_path='faq.json'):
        print("[~] Loading campus chatbot...")

        with open(faq_path, 'r') as f:
            data = json.load(f)

        self.faqs = data.get('faqs', [])

        # Week 4: TF-IDF retriever
        self.retriever = TFIDFFAQRetriever(self.faqs)

        # Week 5: Intent classifier
        self.classifier = IntentClassifier()

        if self.classifier.is_ready():
            print(f"[✓] Intent classifier loaded ({self.classifier.model_type})")
        else:
            print("[!] Intent models not found. Run train_intent.py first.")
            print("    Intent classification will be skipped.")

        print("[✓] Campus Assistance Chatbot ready!\n")

    def get_response(self, user_input):
        user_input = user_input.strip()
        if not user_input:
            return "Please type your question. I'm here to help!"

        # Week 1 — Handle greetings
        greeting_response = handle_greeting(user_input)
        if greeting_response:
            return greeting_response

        # Week 1 — Direct keyword rule match
        rule_response = rule_based_match(user_input)
        if rule_response:
            return rule_response

        # Week 5 — Detect intent
        intent_label = 'unknown'
        intent_conf = 0.0
        if self.classifier.is_ready():
            intent_label, intent_conf = self.classifier.predict(user_input)

        # Week 4 — TF-IDF cosine similarity retrieval (Weeks 2 & 3 used inside)
        answer, score, matched_q = self.retriever.get_best_answer(user_input)

        if answer:
            response = answer
            if intent_label != 'unknown' and intent_conf > 0.4:
                response += f"\n  [Intent: {intent_label} | Confidence: {intent_conf:.0%} | Match score: {score:.2f}]"
            return response

        # Low confidence fallback
        if intent_label != 'unknown' and intent_conf > 0.5:
            return (
                f"I detected your question is about '{intent_label}', but I couldn't find a specific answer. "
                f"Please contact the relevant office or visit portal.campus.edu for more details."
            )

        return (
            "I'm sorry, I didn't understand that. You can ask me about:\n"
            "  • College timings / timetable\n"
            "  • Fees and payments\n"
            "  • Admissions and registration\n"
            "  • Exams and results\n"
            "  • Hostel / accommodation\n"
            "  • Scholarships and financial aid"
        )

    def run(self):
        print("=" * 60)
        print("   🎓 INTELLIGENT CAMPUS ASSISTANCE SYSTEM")
        print("   January Phase — Weeks 1–5 Integrated")
        print("=" * 60)
        print("   Type your question. Type 'bye' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBot: Goodbye! Have a great day! 👋")
                break

            if not user_input:
                continue

            if handle_farewell(user_input):
                print("Bot: Goodbye! Have a great day studying! 👋\n")
                break

            response = self.get_response(user_input)
            print(f"\nBot: {response}\n")


if __name__ == '__main__':
    if not os.path.exists('faq.json'):
        print("[ERROR] faq.json not found. Please run from the january/ directory.")
    elif not os.path.exists('intent_lr.pkl') and not os.path.exists('intent_nb.pkl'):
        print("[WARNING] Intent models not found. Running train_intent.py first...\n")
        import subprocess, sys
        subprocess.run([sys.executable, 'train_intent.py'])
        bot = CampusChatbot()
        bot.run()
    else:
        bot = CampusChatbot()
        bot.run()
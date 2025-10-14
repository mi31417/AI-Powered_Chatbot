from flask import Flask,request,render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('chat.html')

@app.route("/hello/<name>")
def hello(name):
    return f"<p> Hello, {name} !</p>"

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.form.get("message")

    responses = {
    "hi": "Hello! How can I help you today?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! 😊",

    "admission": "Admissions are open for Fall and Spring semesters. You can apply through the Graduate School portal.",
    "courses": "We offer undergraduate and graduate programs in Information Systems and related areas.",
    "library timings": "The library is open from 8 AM to 10 PM on weekdays, and 10 AM to 6 PM on weekends.",
    "myumbc": "Here is the link to myUMBC: https://my.umbc.edu/",
    "assistantship": "You can find assistantship details here: https://informationsystems.umbc.edu/home/graduate-programs/assistantship-opportunities/",
    "contact": "You can contact the department at info@umbc.edu or call (410) 455-3206.",
    }


    for key, value in responses.items():
        if key in user_message.lower():
            reply = value
            break
        else:
         reply = "I'm still learning 😊"
         
    return reply


if __name__ == "__main__":
    app.run(debug=True)

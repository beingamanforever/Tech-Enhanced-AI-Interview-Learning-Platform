from flask import Flask, render_template, request, jsonify, redirect, url_for
from model import generate_output
from faster_whisper import WhisperModel

model_audio = WhisperModel(model_size_or_path="small")

# import google.generativeai as genai
# api_key = ""
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-pro')

app = Flask(_name_)
name = []
position = []
prev_q = []
flag = 0
feedback = ["Grammatical correction here"]
pace  = 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        username = request.form["username"]
        pos= request.form["position"]
        print(username)
        print(pos)
        name.append(username)
        position.append(pos)
        return render_template('home.html',name=name[0])
    return render_template('home.html')

@app.route("/get_flag", methods=["GET"])
def get_flag():
    global feedback
    if len(feedback)>0:
        return jsonify({'flag': feedback[-1]})
    else :
        return jsonify({'flag': ''})

@app.route("/get", methods=["GET", "POST"])
def chat_():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_file.save('audio.wav')
        print("Audio saved successfully")
        return 'Audio uploaded successfully', 200
    else:
        print("Audio not saved")
        return 'No audio file received', 400

DEFAULT_CHUNK_LENGTH = 10

@app.route('/get_text', methods=['GET'])
def get_text():
    global pace
    audio_path = "./audio.wav"
    result = model_audio.transcribe(audio_path)
    segments, info = result
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = ""
    for segment in segments:
        text += segment.text
    pace = calculate_speaking_pace(text, chunk_length=DEFAULT_CHUNK_LENGTH)
    return text

@app.route('/get_pace', methods=['GET'])
def get_pace():
    global pace
    global flag
    output = pace_checker(pace) if flag else ""
    return jsonify({'Pace_Checker': output})


def get_Chat_response(text):
    global flag
    global prev_q
    if flag==0:
        input = f"""    You are now conducting an interview for the {position[0]} role.
  The candidate has responded as follows:
  question : Hello, Tell me about yourself.
  Answer : {text}
  Please formulate a thoughtful follow-up question to probe deeper into the candidate's understanding and experience of the candidate's response in relation to the desired skills and knowledge for the {position[0]} role.
  Keep the response precise. Don't use the word "Great" and be strict.
  output:
  """
        flag += 1
    else :
        input = f"""You are now conducting an interview for the {position[0]} role.
  You have asked the candidate the following question:
  {prev_q[flag-1]}
  
  The candidate has responded as follows:
  {text}
  Please formulate a thoughtful follow-up question to probe deeper into the candidate's understanding and experience of the candidate's response in relation to the desired skills and knowledge for the {position[0]} role.
   Keep the response precise. Don't use the word "Great" and be strict.
  output:
  """
        flag += 1
    # if flag>1:
    #     reponse = answer_checker(text, prev_q[-1])

    output = generate_output(input)
    ##
    feed = grammar_checker(text)
    feedback.append(feed)
    output = output.split(':')[-1]
    prev_q.append(output)
    return output

def calculate_speaking_pace(transcription, chunk_length):
    words = transcription.split()
    num_words = len(words)
    speaking_rate = num_words / chunk_length  # Words per second
    return speaking_rate

def pace_checker(pace):
    optimal_pace_range = (1, 3)
    if optimal_pace_range[0] <= pace <= optimal_pace_range[1]:
        return "Good Pace"
    elif pace < optimal_pace_range[0]:
        return "Very Slow"
    elif pace > optimal_pace_range[1]:
        return "Too Fast"


def grammar_checker(text):
    input = f"""
    Correct “{text}” to standard English and place the results in “Correct Text:”
"""
    output = generate_output(input)
    return output.split(':')[-1]

def answer_checker(text,question):
    # input = f""" Question : {question}
    #             Candidate answer : {text}
    #             Considering the answer for the question, output only 'YES' if answer is correct or else output only 'NO'.
    #         output:
    #  """
    input = f"""### instruction: you are an experienced interviewer.\
   You are interviewing a candidate for the position of {position[0]} .\
   You are tasked to rate an answer provided by the candidate. You should provide a categorical rating and qualitative feedback.\
    The categorical rating should be one of the following values: Good, average, or  Poor.\
      the qualitative feedback should provide sufficient details to justify the categorical rating.\
        the format instructions of the output and the question asked to the candidate and the answer given by the candidate are  given below.\
        "" I  want rating between 1 to 10 only please. 10 showing a perfect answer for the given question. Give rating only only""
        ### question:{question}.\
        ### answer:{text}.\
        ### Rating:
        """
    output = generate_output(input)
    print(output.split("Rating:")[-1])
    return output

if __name__ == '__main__':
    app.run()
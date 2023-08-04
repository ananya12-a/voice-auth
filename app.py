from flask import Flask, render_template, request, redirect
import os

from voice_auth import recognize

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    dist1 = -1
    dist2 = -1
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file1" not in request.files or "file2" not in request.files:
            print("Didn't supply files")
            return redirect(request.url)
        
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        if file1.filename == "" or file2.filename == "":
            print("Didn't upload files")
            return redirect(request.url)
        user = request.form.get('user')
        file1.save(os.path.join('static/voice-samples', 'file1.wav'))
        file2.save(os.path.join('static/voice-samples', 'file2.wav'))
        print(user)
        if file1 and file2:
            dist1 = recognize('static/voice-samples/file1.wav', user)
            dist2 = recognize('static/voice-samples/file2.wav', user)
            print(dist1, dist2)
    return render_template('index.html', score1=dist1, score2=dist2)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8000)
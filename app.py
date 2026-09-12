from flask import Flask, render_template, send_from_directory
import sqlite3
import os

app = Flask(__name__)

# STEP 2 FIX → tell Flask where images are stored
EVIDENCE_FOLDER = os.path.join(os.getcwd(), "evidence")
app.config["EVIDENCE_FOLDER"] = EVIDENCE_FOLDER


def get_data():
    conn = sqlite3.connect("attendance.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 50")
    rows = cur.fetchall()
    conn.close()
    return rows


@app.route("/")
def home():
    rows = get_data()
    return render_template("index.html", rows=rows)


# STEP 2 FIX → this route serves the image file
@app.route("/evidence/<path:filename>")
def evidence_file(filename):
    return send_from_directory(app.config["EVIDENCE_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)


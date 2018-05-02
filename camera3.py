from flask import Flask, render_template, request
from picamera import PiCamera
import face_detect2

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template("main.html")

@app.route('/photo', methods=['POST'])
def photo_page():
    num_photos = int(request.form['num_photos'])

    check=True
    camera=PiCamera()
    camera.resolution=(640, 480)
    for i in range(num_photos):
        camera.capture("static/image_{}.jpg".format(i))
        check=face_detect2.facedetect(i)
    camera.close()

    return render_template("photo.html", num_photos=num_photos, check=check)

if __name__=='__main__':
    app.run(host="0.0.0.0")

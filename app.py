from flask import Flask, render_template, redirect, request
import caption

app = Flask(__name__)

@app.route('/')

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def img_caption():

    if request.method == "POST":

        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)

        image_caption = caption.caption_for_the_image(path)
        print(image_caption)

        result_dict = {
            'image' : path,
            'caption' : image_caption
        }

    return render_template("index.html", your_result=result_dict)

@app.route('/about.html')
def about():
    images = ["./static/i1.jpg", "./static/i2.jpg"]
    return render_template("about.html", img_paths=images)

@app.route('/home')
def home():
    return redirect('/')

if __name__ == '__main__':

    app.run(debug=True)
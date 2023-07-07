import os
import sys

sys.path.append('ms_istft_vits')
import requests
from flask import (
    Flask,
    request,
    send_file,
    render_template,
    jsonify,
    Response,
    redirect,
    url_for,
)
from io import BytesIO
import scipy.io.wavfile as swavfile
from synthesis import generate_audio, SAMPLE_RATE

app = Flask(__name__)


@app.after_request
def allow_cors(response):
    response.headers['Access-Control-Allow-Origin'] = "*"
    return response


@app.route("/")
def index():
    return redirect(url_for("open_captions_overlay"))


@app.route("/tts-server/cc-overlay")
def open_captions_overlay():
    return render_template("cc-overlay.html")



@app.route("/tts-server/api/vits")
def infer_vits():
    text = request.args.get("text", "")

    if not text:
        return "text shouldn't be empty", 400

    wav = BytesIO()
    try:
        audio = generate_audio(text)
        swavfile.write(wav, rate=SAMPLE_RATE, data=audio)

    except Exception as e:
        return f"Cannot generate audio: {str(e)}", 500

    return send_file(wav, mimetype="audio/wave", attachment_filename="audio.wav")


@app.route("/favicon.ico")
def favicon():
    return "I don't have favicon :p", 404


@app.route("/<path:path>")
def twip_proxy(path):
    new_url = request.url.replace(request.host, "twip.kr")
    resp = requests.request(
        method=request.method,
        url=new_url,
        headers={key: value for (key, value) in request.headers if key != "Host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
    )
    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    headers = [
        (name, value)
        for (name, value) in resp.raw.headers.items()
        if name.lower() not in excluded_headers
    ]
    content = resp.content
    if new_url.startswith("http://twip.kr/assets/js/alertbox/lib-"):
        content = (
            resp.text
            + f"""
        const original_function = Howl.prototype.init;
        Howl.prototype.init = function (o) {{
            if (o.src.startsWith('https://www.google.com/speech-api/v1/synthesize?text=')) {{
                o.src = o.src.replace(
                    'https://www.google.com/speech-api/v1/synthesize?text=',
                    '/tts-server/api/vits?text='
                );
                o.html5 = false;
                o.volume = o.volume * 2;
            }}
            return original_function.call(this, o);
        }}
        """
        )
    response = Response(content, resp.status_code, headers)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.environ.get("TTS_DEBUG", "0") == "1")

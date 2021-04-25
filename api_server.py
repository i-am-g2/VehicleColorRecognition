import argparse
import io
from PIL import Image
from infer import infer 
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def image_handler():
	bio = io.BytesIO()
	request.files['image'].save(bio)
	image = Image.open(bio)
	result = infer(image)
	return jsonify({'result': result}) 

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', default = 8769 , type = int)
	args = parser.parse_args()
	app.debug = True
	app.run('0.0.0.0', args.port)



if __name__ == "__main__":
	main()
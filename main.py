import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from dotenv import load_dotenv
from src.model import ClimbNet
from src.generator import ClimbGenerator
from src.utils import OutputConversion

load_dotenv()
app = Flask(__name__, template_folder='templates')
CORS(app)
selected_positions = []


@app.route('/')
def index():
    return render_template('home+.html')


@app.route('/classification')
def classification():
    return render_template('classification.html')


@app.route('/generator')
def generator():
    input_grade = request.args.get('grade')
    grade_to_level = {
        'V4': 'easy', 'V5': 'easy',
        'V6': 'medium', 'V7': 'medium',
        'V8': 'hard', 'V9': 'hard',
        'V10': 'very_hard', 'V10+': 'very_hard', 'V11':
        'very_hard', 'V12': 'very_hard', 'V13': 'very_hard'
    }

    level = grade_to_level[input_grade]

    generator = ClimbGenerator(level)
    clf = ClimbNet()
    weights_filepath = "src/climbnet_weights.pth"
    clf.load_state_dict(torch.load(weights_filepath))

    clf.eval()
    while True:
        climb, labels, vector = generator.generate_climb()
        vector = torch.tensor(vector).to(torch.float32).flatten()
        with torch.no_grad():
            soft_pred = clf(vector)
        hard_pred = int(soft_pred.argmax())
        out_conv = OutputConversion()
        predicted_grade = out_conv.convert(hard_pred)

        if predicted_grade == input_grade:
            break
    return render_template('generator.html')

@app.route('/api/save-positions', methods=['POST'])
def save_positions():
    global selected_positions
    global grade_output
    data = request.json
    selected_positions = data

    indices = list()
    for coord_dict in selected_positions:
        row = coord_dict['row']
        col = coord_dict['col']
        idx = ((18 - row - 1)*11) + (col)
        indices.append(idx)

    position_ohe = np.zeros(198)
    for idx in indices: 
        position_ohe[idx] += 1

    model_input = torch.tensor(position_ohe).to(torch.float32).flatten() 

    model = ClimbNet()
    weights_filepath = "src/climbnet_weights.pth"
    model.load_state_dict(torch.load(weights_filepath))

    model.eval()
    with torch.no_grad():
        soft_pred = model(model_input)    
    hard_pred = int(soft_pred.argmax())
    out_conv = OutputConversion()
    grade_output = out_conv.convert(hard_pred)
    # print('Received positions:', selected_positions)
    # print(position_ohe)
    print(grade_output)

    return render_template('classification.html', pred=grade_output)  # jsonify({'prediction': grade_output}) #
    # return jsonify({'status': 'success', 'data': selected_positions})


@app.route('/api/get-positions', methods=['GET'])
def get_positions():
    global selected_positions
    return jsonify({'status': 'success', 'data': selected_positions})


@app.route('/prediction')
def get_prediction():
    global grade_output
    return render_template('classification.html', pred=grade_output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5005)

import matplotlib
matplotlib.use('Agg') 
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from dotenv import load_dotenv
from src.model import ClimbNet
from src.generator import ClimbGenerator
from src.utils import OutputConversion
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
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

@app.route('/generator', methods=['GET', 'POST'])
def generator():
    options = ['V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
           'V10', 'V10+']
    grade = None 
    vector = None
    input_grade = None
    selected_cells = None
    image = None
    if request.method == 'POST':
        if 'grade_dropdown' in request.form:
            input_grade = request.form['grade_dropdown']
    if input_grade:  # Only proceed if input_grade is not None
        grade_to_level = {
            'V4': 'easy', 'V5': 'easy',
            'V6': 'medium', 'V7': 'medium',
            'V8': 'hard', 'V9': 'hard',
            'V10': 'very_hard', 'V10+': 'very_hard'
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
                print(f"predicted_grade:{predicted_grade}")
                break
            
            
            x_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            y_labels = [str(i) for i in range(1, 19)]

            # List the selected cells
            selected_cells = []
            for idx, value in enumerate(vector):
                if value == 1.0:
                    row = idx // 11
                    col = idx % 11
                    label = f'{x_labels[col]}{y_labels[row]}'
                    selected_cells.append(label)
            image = plot_generated_climb(climb,labels)
            
            
    return render_template('generator.html', grade=input_grade, vec=selected_cells,img_path=image)#)input_grade=input_grade)






def plot_generated_climb(climb, labels, img_path='static/generate.jpeg'):
    
    h_incpt = 93.3
    h_scalar = 50
    v_incpt = 63.5
    v_scalar = 50
    holds = []
    for row in range(18):
        for col in range(11):
            hold_id = f"{chr(65 + col)}{row + 1}"
            x = h_incpt + h_scalar * col
            y = v_incpt + v_scalar * row
            holds.append({"id": hold_id, "x": x, "y": y})
    G = nx.Graph()
    for hold in holds:
        G.add_node(hold['id'], pos=(hold['x'], hold['y']))
    
    wall_img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(wall_img, extent=[0, wall_img.shape[1], 0, wall_img.shape[0]])
    # Get positions of holds
    pos = {hold['id']: (hold['x'], hold['y']) for hold in holds}
    # Highlight the generated climb

    for hold, label in zip(climb, labels):
        print(climb, labels)
        color = 'green' if label == 'starting hold' else ('blue' if label == 'intermediate hold' else 'red')
        nx.draw_networkx_nodes(G, pos, nodelist=[hold], ax=ax, node_size=600, node_color='none', edgecolors=color, linewidths=2.5)
        # nx.draw_networkx_nodes(G, pos, nodelist=[hold], ax=ax, node_size=600, node_color=color, edgecolors='none', linewidths=2.5, alpha=0.5)
    output_image_path = 'static/generated_climb.png'
    plt.savefig('static/generated_climb.png')  # Save the image to a file
    plt.close()
    return output_image_path



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

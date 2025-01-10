from flask import Flask, request, jsonify
from EightPuzzle import EightPuzzle, Agent

app = Flask(__name__)

# Scramble
@app.route('/scramble', methods=['POST'])
def scramble():
    pass

# Returns the three solutions to the 
@app.route('/solve', methods=['POST'])
def post_solution():
    pass


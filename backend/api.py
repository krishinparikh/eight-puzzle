from flask import Flask, request, jsonify
from flask_cors import CORS

from EightPuzzle import EightPuzzle, Agent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/solve', methods=['POST'])
def solve_puzzle():
    data = request.get_json()
    puzzleState = data.get('puzzle')

    solutions = [
        Agent.solveBFS(EightPuzzle(puzzleState), maxnodes=10000000),
        Agent.solveAstar(EightPuzzle(puzzleState), "h1", maxnodes=10000000),
        Agent.solveAstar(EightPuzzle(puzzleState), "h2", maxnodes=10000000)
    ]

    return jsonify({"solutions": solutions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
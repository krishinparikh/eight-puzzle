// Import necessary libraries
import React, { useState, useEffect } from 'react';
import './App.css';

// Define the initial 8-puzzle state
const initialPuzzle = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8], // 0 represents the blank space
];

const App: React.FC = () => {
  const [puzzle, setPuzzle] = useState(initialPuzzle);
  const [blankPosition, setBlankPosition] = useState({ row: 0, col: 0 });
  const [solutions, setSolutions] = useState<string[] | null>(null);

  // Handle key presses to manipulate the puzzle
  const handleKeyDown = (event: KeyboardEvent) => {
    event.preventDefault(); // Prevent scrolling on arrow key press

    const { row, col } = blankPosition;
    let newPuzzle = [...puzzle.map(row => [...row])];

    switch (event.key) {
      case 'ArrowUp':
        if (row < 2) {
          [newPuzzle[row][col], newPuzzle[row + 1][col]] = [newPuzzle[row + 1][col], newPuzzle[row][col]];
          setBlankPosition({ row: row + 1, col });
        }
        break;
      case 'ArrowDown':
        if (row > 0) {
          [newPuzzle[row][col], newPuzzle[row - 1][col]] = [newPuzzle[row - 1][col], newPuzzle[row][col]];
          setBlankPosition({ row: row - 1, col });
        }
        break;
      case 'ArrowLeft':
        if (col < 2) {
          [newPuzzle[row][col], newPuzzle[row][col + 1]] = [newPuzzle[row][col + 1], newPuzzle[row][col]];
          setBlankPosition({ row, col: col + 1 });
        }
        break;
      case 'ArrowRight':
        if (col > 0) {
          [newPuzzle[row][col], newPuzzle[row][col - 1]] = [newPuzzle[row][col - 1], newPuzzle[row][col]];
          setBlankPosition({ row, col: col - 1 });
        }
        break;
    }

    setPuzzle(newPuzzle);
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [blankPosition, puzzle]);

  // Render the puzzle as a grid
  return (
    <div className="app-container">
      <header className="app-header" style={{ paddingLeft: '20px' }}>
        <h1>8-Puzzle Solver</h1>
        <p>
          Play with an online version of the famous 8-Puzzle using your arrow keys. Click "Solve" to find and compare three different
          algorithmic solutions to your scrambled puzzle. To learn more, please visit this project's <a href="https://github.com/krishinparikh/eight-puzzle" target="_blank" rel="noopener noreferrer">GitHub repository</a>. Enjoy!
        </p>
      </header>
      <div className="puzzle-and-solutions">
        <div className="puzzle-container">
          {puzzle.map((row, rowIndex) => (
            <div key={rowIndex} className="puzzle-row">
              {row.map((cell, colIndex) => (
                <div key={colIndex} className={`puzzle-cell ${cell === 0 ? 'blank' : ''}`}>
                  {cell !== 0 && cell}
                </div>
              ))}
            </div>
          ))}
          <button
            className="solve-button"
            onClick={async () => {
              try {
                const response = await fetch('https://eight-puzzle-backend.onrender.com/solve', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ puzzle }),
                });
                if (response.ok) {
                  const data = await response.json();
                  setSolutions(data.solutions); // Assuming the API sends an array of solutions
                } else {
                  console.error('Failed to fetch solutions');
                }
              } catch (error) {
                console.error('Error:', error);
              }
            }}
          >
            Solve
          </button>
        </div>
        {solutions && (
          <div className="solutions-grid">
            {solutions.map((solution, index) => (
              <div key={index} className="solution-column">
                <h4>
                  {index === 0
                    ? 'Breadth First Search Solution'
                    : index === 1
                    ? 'A* Search (Heuristic 1) Solution'
                    : 'A* Search (Heuristic 2) Solution'}
                </h4>
                <pre>{solution}</pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;

import * as sm from '@shumai/shumai';

// Define a function to handle user input and model response
async function chatWithData() {
  const model = sm.network.remote_model('http://localhost:3000');
  const input = sm.tensor([1, 2, 3]); // Replace with your input data
  const response = await model(input);
  console.log('Model response:', response.toFloat32Array());
}

// Serve the model in the terminal
sm.network.serve_model(chatWithData);

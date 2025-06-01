// start.js
require('dotenv').config();
const { exec } = require('child_process');

const FRONT_PORT = process.env.FRONT_PORT || 3000;

exec(`cross-env PORT=${FRONT_PORT} react-scripts start`, {
  stdio: 'inherit'
});
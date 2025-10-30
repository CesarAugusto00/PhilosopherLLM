# Simple Chat Echo Bot

A simple Flask web application that creates a chat interface where messages are echoed back to the user.

## Features

- Clean, responsive Bootstrap UI
- Real-time message echoing
- Timestamped messages
- User and bot message differentiation
- Mobile-friendly design

## Setup and Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   ```
   http://localhost:5000
   ```

## How it works

- Type a message in the input field and press Enter or click Send
- The message will appear on the right side (user message)
- The bot will immediately echo the message back on the left side
- All messages are timestamped and stored in memory

## Files

- `app.py` - Flask application with API endpoints
- `templates/index.html` - HTML template with Bootstrap styling and JavaScript
- `requirements.txt` - Python dependencies


# Sign Language Detection Project

This project is a Sign Language Detection app that translates sign language gestures into text and speech. It uses a CNN model with a Flask backend for easy deployment and interaction.

## Prerequisites
- Python 3.10
- Git
- Docker (optional, if using Docker setup)
- Virtual Environment (optional, if using Python setup)

## Getting Started

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/nadaYossef/graduation-project-sign-language-detection.git
cd graduation-project-sign-language-detection
```

### 2. Setting Up the Python Environment

#### Using Virtual Environment on Windows
1. Create the virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

#### Using Virtual Environment on macOS
1. Create the virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Running the Application Locally
Once the environment is set up, start the application:
```bash
python app.py
```

The application should now be accessible in your browser at `http://127.0.0.1:5000`.

## Using Docker
Alternatively, you can use Docker for easier setup.

1. Make sure Docker is running on your machine.
2. Run the following command to build and start the containers:
   ```bash
   docker-compose up --build
   ```

This will launch the application in a Docker container, accessible at `http://127.0.0.1:5000`.

for train dataset download from here : https://drive.google.com/file/d/1sWNJYmKZyHWEwucrfZ1JmR0RiWwUhq26/view?usp=sharing

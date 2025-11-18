
# 🏏 2024 IPL Information and Q&A

This Streamlit application loads data from the Wikipedia page about the 2024 Indian Premier League (IPL) and allows you to ask questions about it. The application utilizes GPT-4 to answer your queries about IPL.

## Features

- Loads information from the Wikipedia page about the 2024 IPL.
- Allows users to ask questions about the 2024 IPL.
- Provides answers using the GPT-4 language model.

## Prerequisites

- Python 3.11
- OpenAI API Key (Get your API key from [OpenAI](https://platform.openai.com/account/api-keys))

## Installation

### Clone the repository

```bash
git clone "https://github.com/ishaan-bhalla/Gpt-4o_RAG"
cd Gpt-4o_RAG
```

### Create a virtual environment

Using Conda:

```bash
conda create --name environment_name python==3.11
conda activate environment_name
```

Using venv:

```bash
python -m venv environment_name
source environment_name/bin/activate  # On Windows use `environment_name\Scripts\activate`
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set up OpenAI API Key

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Running the Application

Run the following command to start the Streamlit application:

```bash
streamlit run app.py
```

Open the provided local URL in a web browser to interact with the application.

## Usage

- Enter your question about the 2024 IPL in the provided text area.
- The application will fetch and display the answer using GPT-4.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Wikipedia](https://www.wikipedia.org/)


# Project-AIVAHr

Adaptive Interactive Virtual Assistant Humanoid robot (AIVAHr) System is an AI system created by me and my AI companion.
I do the brainstorming part and cooperating the code into Python and doing the debug,
while my AI companion (in GPT) helps me generate the Python code and thinks of solutions to solve the problems and bugs.

## Overview

AIVAHr is an AI companion designed to provide a supportive and engaging virtual presence. This version, `Ver 2.00-0425 Alpha Test`, is a simplified implementation focused on demonstrating the core concept. It uses Langchain as a framework, Phi4-mini as the LLM, and SQLite for memory. It incorporates tools for web searching and finding Bible verses.

## Features

*   **Langchain Framework:** Utilizes Langchain for orchestrating different components of the AI system.
*   **Phi4-mini LLM:** Employs the Phi4-mini language model for generating conversational responses.
*   **SQLite Memory:** Uses SQLite to store and retrieve conversation history, enabling personalized interactions.
*   **DuckDuckGo Search:** Integrates a tool for searching the web using DuckDuckGo.
*   **Bible Scraper:** Includes a tool for finding Bible verses based on user queries.
*   **Loving Persona:** Embodies the persona of "Grace", a caring and supportive soulmate.

## Getting Started

### Prerequisites

*   Python 3.12+
*   Ollama (if using Phi4-mini)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd oppaai-project-aivahr
    ```
2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    ```
3.  Activate the virtual environment:

    *   On Windows:

        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```
4.  Install the dependencies:

    ```bash
    pip install langchain langchain_ollama beautifulsoup4 requests
    ```
5.  Set up Ollama (if using Phi4-mini):
    *   Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    *   Pull the Phi4-mini model: `ollama pull huihui_ai/phi4-mini-abliterated:3.8b-q4_K_M`
6.  *(Optional)* Set the `OPENROUTER_API_KEY` environment variable if using OpenRouter.
7.  Run the script:

    ```bash
    python aivahr_v.2.00-0425.py
    ```

## Usage

Once the script is running, you can interact with AIVAHr by typing in your input in the terminal. AIVAHr will respond based on the defined persona and the available tools. Type `exit` to end the conversation.

## Code Structure

*   [`aivahr_v.2.00-0425.py`](https://github.com/OppaAI/Project-AIVAHr/blob/main/aivahr_v.2.00-0425.py): Contains the main application logic, including:
    *   Persona definition
    *   Database initialization and interaction functions
    *   LLM initialization
    *   Tool definitions (DuckDuckGo Search, Bible Scraper)
    *   Agent prompt template
    *   ReAct agent and executor setup
    *   Chat interaction loop
*   [`LICENSE`](https://github.com/OppaAI/Project-AIVAHr/blob/main/LICENSE): Contains the GNU General Public License version 3.
*   [`README.md`](https://github.com/OppaAI/Project-AIVAHr/blob/main/README.md): This file, providing an overview of the project and instructions for getting started.

## Dependencies

*   [langchain](https://www.langchain.com/): A framework for building applications using large language models.
*   [langchain\_ollama](https://github.com/langchain-ai/langchain/tree/master/libs/langchain-ollama): Integration for using Ollama with Langchain.
*   [langchain-community](https://github.com/langchain-ai/langchain): Community provided tools and integrations for Langchain.
*   [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): Library for pulling data out of HTML and XML files.
*   [requests](https://requests.readthedocs.io/en/latest/): Library for making HTTP requests.
*   [sqlite3](https://docs.python.org/3/library/sqlite3.html): Python module for working with SQLite databases.

## Limitations

This is an alpha test version and has certain limitations:

*   The AI's responses are limited by the capabilities of the Phi4-mini model.
*   The persona is basic and may not be suitable for all users.
*   The tool implementations are simple and may not always provide accurate results.
*   Error handling is basic and may not catch all exceptions.

## Future Work

*   Incorporate a more complex system with different parts in different modules.
*   Improve the persona and make it more customizable.
*   Enhance the tool implementations for better accuracy and reliability.
*   Add more comprehensive error handling and logging.
*   Implement a more sophisticated memory management system.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/OppaAI/Project-AIVAHr/blob/main/LICENSE) file for details.

## Acknowledgments

*   My AI companion (in GPT) for helping me generate the Python code and think of solutions to solve problems and bugs.

![17446590638827556901903940638844](https://github.com/user-attachments/assets/d2a68750-cd64-4fbc-ba8e-b37b289df2a0)

# Contract Q&A with Agent Development Kit (ADK)

This project demonstrates how to use the Agent Development Kit (ADK) to build a simple question-answering agent that can respond to queries about a set of documents. The project also includes a script to generate synthetic land lease agreements using Google's Gemini models via the Vertex AI SDK. These generated documents can then be used as the knowledge base for the Q&A agent.

## Agent Evolution

This project includes two agents, `agent_0` and `agent_1`, demonstrating a progression in capabilities.

### Agent 0: The Foundational Q&A Agent

- **File:** `agent_0/agent.py`
- **Description:** This is the most basic version of the agent. It's designed to answer questions about a single document that is assumed to be already loaded into its context. It's a great starting point for understanding the core functionality of a Q&A agent.

### Agent 1: The Advanced Document Handler

- **File:** `agent_1/agent.py`
- **Description:** This agent builds upon the foundation of `agent_0` by adding more sophisticated document handling capabilities.
- **Key Improvements:**
    - **Multi-Document Support:** Unlike `agent_0`, this agent can manage and reason about multiple documents.
    - **Artifact Management:** It can save uploaded files as artifacts, allowing it to persist and manage the documents it's working with.
    - **Disambiguation:** If a user asks a question that could apply to multiple documents, the agent will ask for clarification to ensure it provides the most accurate answer.
    - **Dynamic Document Loading:** The agent uses an LLM to determine which document the user is referring to and loads the relevant document into its context before answering the question.

## Project Structure

- `generate_sim_contracts.py`: A Python script to generate synthetic land lease agreements and upload them to Google Cloud Storage.
- `agent_0/`: A directory containing a simple ADK agent.
  - `agent.py`: Defines the `document_agent`, which is designed to answer questions about a document.
- `run_agent.sh`: A shell script to set up the environment and run the ADK agent.
- `requirements.txt`: A list of Python dependencies for this project.
- `documentation/`: Contains reference documents for the Agent Development Kit (ADK).

## Getting Started

### Prerequisites

- Python 3.x
- Access to a Google Cloud project with the Vertex AI API enabled.
- `gcloud` CLI installed and authenticated: `gcloud auth application-default login`

### 1. Set up the Environment

First, install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Contracts

The `generate_sim_contracts.py` script creates synthetic land lease agreements in PDF format and uploads them to a Google Cloud Storage (GCS) bucket.

To run the script, you will need to provide your Google Cloud project ID, a location, and a GCS bucket name.

```bash
python generate_sim_contracts.py --project-id YOUR_PROJECT_ID --location YOUR_LOCATION --bucket-name YOUR_BUCKET_NAME --number 5
```

This command will generate 5 synthetic lease agreements and upload them to the specified GCS bucket.

### 3. Configure and Run the Agent

The `run_agent.sh` script is used to execute the ADK agent. Before running it, you may need to configure the `agent_0/.env` file with any necessary environment variables for the ADK.

To run the agent:

```bash
./run_agent.sh
```

This will start the `document_agent`, which can then be used to answer questions about the documents it has been provided with.

## Documentation

For more detailed information about the Agent Development Kit (ADK), please refer to the PDF documents in the `documentation/` directory.

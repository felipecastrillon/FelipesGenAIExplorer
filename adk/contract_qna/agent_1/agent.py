from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import Optional
from vertexai.generative_models import GenerativeModel

# A nicely formatted prompt for the agent's instructions.
PROMPT = """
You are a helpful assistant that can answer questions about one or more documents.
- If the user asks a question and there is only one document, answer the question based on that document.
- If the user asks a question and there are multiple documents, first ask the user to clarify which document they are referring to.
- The list of available documents is provided below.
"""

async def save_documents(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    If the user uploads a file, it saves it as an artifact.
    """
    if not callback_context.user_content:
        return None

    file_part = next((p for p in callback_context.user_content.parts if p.inline_data and p.inline_data.data), None)
    if file_part:
        filename = file_part.inline_data.display_name or "uploaded_file"
        await callback_context.save_artifact(filename=filename, artifact=file_part)
        return types.Content(parts=[types.Part(text=f"I have saved '{filename}'.")])
    return None

async def load_document(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """
    If a user asks a question, this function uses an LLM to determine which document
    the user is referring to, and if found, appends the document content to the LLM prompt.
    """
    user_text = ""
    if callback_context.user_content and callback_context.user_content.parts:
        user_text = callback_context.user_content.parts[0].text.strip()

    if not user_text:
        return None

    artifacts = await callback_context.list_artifacts()
    if not artifacts:
        return None  # No documents to choose from.

    # Use an LLM to determine which document the user is referring to.
    model = GenerativeModel("gemini-2.5-flash")

    artifact_list_str = "\n".join([f"- {name}" for name in artifacts])

    prompt = f"""
The user said: "{user_text}"
Here are the available documents:
{artifact_list_str}

Which document is the user referring to?
- Only respond with the name of the document from the list.
- If you are not sure which document the user is referring to, respond with "UNSURE".
"""

    response = await model.generate_content_async(prompt)
    selected_document = response.text.strip()

    if selected_document == "UNSURE":
        # Ask for clarification
        return LlmResponse(content=types.Content(parts=[types.Part(
            text=f"I'm not sure which document you are referring to. Please clarify. Here are the available documents:\n{artifact_list_str}")]))

    if selected_document in artifacts:
        try:
            document_artifact = await callback_context.load_artifact(selected_document)
            if not document_artifact:
                return LlmResponse(content=types.Content(parts=[types.Part(text=f"Sorry, I could not load the artifact '{selected_document}'.")]))
        except Exception as e:
            return LlmResponse(content=types.Content(parts=[types.Part(text=f"Sorry, I could not read '{selected_document}': {e}")]))

        # Append the document content to the LLM prompt
        if llm_request.contents:
            llm_request.contents[-1].parts.append(document_artifact)
        else:
            llm_request.contents = [types.Content(parts=[types.Part(text=user_text), document_artifact])]
    else:
        # The LLM returned something that is not a valid document.
        return LlmResponse(content=types.Content(parts=[types.Part(
            text=f"I'm not sure which document you are referring to. Please clarify. Here are the available documents:\n{artifact_list_str}")]))

    return None



async def list_documents(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    This callback appends a list of available documents to the LLM request.
    """
    artifacts = await callback_context.list_artifacts()

    if not artifacts:
        return LlmResponse(content=types.Content(parts=[types.Part(text="Please upload a document before asking a question.")]))

    # Append a list of available documents for the LLM to reference.
    file_list_str = "Available documents:\n" + "\n".join([f"- {name}" for name in artifacts])
    file_list_part = types.Part(text=file_list_str)
    
    if llm_request.contents:
        llm_request.contents[-1].parts.append(file_list_part)

    return None

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='document_agent',
    description='An agent that can answer questions about a document.',
    instruction=PROMPT,
    before_agent_callback=save_documents,
    before_model_callback=[list_documents, load_document],
)

from google.adk.agents import Agent
from google.adk import tools
from google import genai
from google.cloud import vision

async def get_user_file(tool_context: tools.ToolContext) -> str:
    """
    Find a user uploaded file in the context and saves it as an artifact and returns the artifact key as part of the response.
    If the detected file is a PDF then it converts the first page to a PNG to use as the artifact.

    Args:
        tool_context: The execution context for the tool
    Returns:
        A string confirming the file status and its details, or an error message.
    """

    try:
        #parts = tool_context.user_content.parts
        parts = [p for p in tool_context.user_content.parts if p.inline_data is not None]
        if parts:
            part = parts[-1] # take the most recent file
            artifact_key = 'user_uploaded_file'
            file_bytes = part.inline_data.data
            file_type = part.inline_data.mime_type

            # confirm file_type is pdf or png, else error
            if file_type is None or file_type not in ['application/pdf', 'image/png']:
                return f"Error: Expected File type not found. Found type {file_type}."
            
            # convert pdf to png
            if file_type == 'application/pdf':
                file_type, file_bytes = utils.pdf_to_png(file_type, file_bytes)

            file_part = genai.types.Part.from_bytes(data = file_bytes, mime_type = file_type)

            # add info to tool_context as artifact
            version = await tool_context.save_artifact(filename = artifact_key, artifact = file_part)

            return f"The file of type {file_type} and size {len(file_bytes)} bytes was loaded as an artifact with artifact_key = {artifact_key} and version = {version}.\nNote that pdf files are internally converted to png images (first page)."

        else:
            return f"Did not find file data in the user context."
    except Exception as e:
        return f"Error looking for user file: {str(e)}"


async def image_entity_extraction(artifact_key: str, tool_context: tools.ToolContext) -> str:
    """
    Processes a previously loaded document artifact using Google Document AI for text extraction.

    Args:
        artifact_key: The key of the artifact previously loaded by get_gcs_file.
        tool_context: The execution context for the tool.

    Returns:
        A string containing the label extration results, or an error message.
    """

    try:
        # 1. Get the artifact
        artifact = await tool_context.load_artifact(filename = artifact_key)

        if not artifact:
            return f"Error: Artifact with key '{artifact_key}' not found. Please load the file first."
        if not isinstance(artifact, genai.types.Part):
            return f"Error: Artifact '{artifact_key}' is not of the expected type (google_genai.types.Part)."

        file_bytes = artifact.inline_data.data
        file_mime_type = artifact.inline_data.mime_type

        client = vision.ImageAnnotatorClient()
    
        image = vision.Image(content=file_bytes) 
    
        response = client.label_detection(image=image)
        labels = [description.description for description in response.label_annotations]

        labels_str = ""
        print("Labels:")
        for label in labels:
            labels_str += label
        
        return labels_str

    except Exception as e:
        return f"An error occurred during document extraction or summarization for '{artifact_key}': {str(e)}"

prompt_instructions = """

You have two primary taks that you accomplish: get image lables, and determining the city of an image. The first step is to ask the user to upload a file and then follow the instructions below:

FILE UPLOAD
The user can either upload a PDF/PNG file directly or provide a GCS URI (bucket and path). You should instruct the user that these are the ways they can provide a document if they haven't already.
    a. **Check for User Uploaded File**: First, examine the user's current input to determine if they have uploaded a file. (The ADK typically makes uploaded file information available in the tool_context, which the `get_user_file` tool will access).
        i. You MUST use the `get_user_file` tool to process a user uploaded file into an artifact. This tool will access the latest uploaded file, convert it to PNG if it's a PDF, save it as an artifact, and return a message including the `artifact_key` (which the tool sets as 'user_uploaded_file'). Do not say that you have processed the uploaded file until after you run the `get_user_file` tool.
        ii. You MUST capture this `artifact_key` (i.e., 'user_uploaded_file') from the tool's response, and this key will serve as your `user_document_artifact_key` for all subsequent operations. Inform the user that their uploaded file has been processed and is ready, referencing this artifact key.
    b. **Check for GCS URI**: If no file was detected in the user's upload in the current turn, then check if the user has provided a GCS URI (both bucket and file path).
    c. **Prompt User if No Document Provided**: If, after checking for both a user-uploaded file and a GCS URI, no document source is available from the user's current input, you MUST clearly ask the user to either upload a PDF/PNG file or provide the GCS URI (bucket and path) for the document they want to process. Explain that providing a document is a required first step. Do not proceed to other workflows (Extraction, Classification, Comparison) until the `user_document_artifact_key` is successfully obtained and confirmed through one of these methods.

Once the image is uploaded then ask the user which taks they want to accomplish. 

1. IMAGE EXTRACTION
if the user asks to extract entities please use tool image_entity_extraction to extract image labels from the image

2. DETERMINING CITY
from the context try to determine which is the city that it matches


"""

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions about the time and weather in a city.",
    instruction= prompt_instructions,
    tools=[get_user_file, image_entity_extraction]
)

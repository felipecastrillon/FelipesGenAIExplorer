from google.adk.agents import Agent
from google.adk import tools
from google import genai
from google.cloud import vision
# import utils # Assuming 'utils.py' contains the pdf_to_png function

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
        parts = [p for p in tool_context.user_content.parts if p.inline_data is not None]
        if parts:
            part = parts[-1] # take the most recent file
            artifact_key = 'user_uploaded_file'
            file_bytes = part.inline_data.data
            file_type = part.inline_data.mime_type

            # confirm file_type is pdf or png, else error
            if file_type is None or file_type not in ['application/pdf', 'image/png']:
                return f"Error: Unsupported file type. Please upload a PNG or PDF file. Found type: {file_type}."
            
            # convert pdf to png
            if file_type == 'application/pdf':
                # Assuming a utility function `pdf_to_png` exists. Make sure it's imported.
                file_type, file_bytes = utils.pdf_to_png(file_type, file_bytes)

            file_part = genai.types.Part.from_bytes(data = file_bytes, mime_type = file_type)

            # add info to tool_context as artifact
            version = await tool_context.save_artifact(filename = artifact_key, artifact = file_part)

            return f"The file was successfully processed and saved as artifact '{artifact_key}' (version: {version}).\nNote: PDF files are converted to a single PNG image of the first page."

        else:
            return f"Did not find file data in the user context."
    except Exception as e:
        return f"Error looking for user file: {str(e)}"


async def image_entity_extraction(artifact_key: str, tool_context: tools.ToolContext) -> str:
    """
    Processes a previously loaded image artifact using Google Cloud Vision for label detection.

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

        if not labels:
            return "No labels were extracted from the image."

        return "Extracted labels: " + ", ".join(labels)

    except Exception as e:
        return f"An error occurred during label extraction for artifact '{artifact_key}': {str(e)}"

prompt_instructions = """
You are an intelligent agent that analyzes images. Your primary tasks are to extract labels from an image and to try and determine the city depicted in an image.

Your first step is always to get an image from the user.

FILE UPLOAD
1.  **Check for User Uploaded File**: First, check if the user has uploaded a file.
    - If they have, you MUST use the `get_user_file` tool to process it. This tool saves the file as an artifact with the key 'user_uploaded_file'.
    - After the tool runs successfully, confirm to the user that the file has been processed and is ready for analysis.
2.  **Prompt User if No File Provided**: If the user has not uploaded a file, you MUST ask them to upload a PNG or PDF image. Do not proceed until a file is provided and processed by the `get_user_file` tool.

Once the image is successfully uploaded and processed, ask the user which task they want to perform.

AVAILABLE TASKS
1.  **Image Label Extraction**: If the user asks to extract labels, entities, or describe the image, use the `image_entity_extraction` tool with the artifact key 'user_uploaded_file'.
2.  **Determine City**: If the user asks to identify the city in the image, use the extracted labels and your own knowledge to deduce the location. You do not have a specific tool for this; you must reason based on the visual information.
"""


root_agent = Agent(
    name="image_analysis_agent",
    model="gemini-1.5-flash",
    description="An agent that analyzes images to extract labels and identify cities.",
    instruction= prompt_instructions,
    tools=[get_user_file, image_entity_extraction]
)

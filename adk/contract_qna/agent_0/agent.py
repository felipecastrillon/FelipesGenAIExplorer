from google.adk.agents import Agent

# The DocumentAgent will process the document and answer questions.
root_agent = Agent(
    model='gemini-2.5-flash',
    name='document_agent',
    description='An agent that can answer questions about a document.',
    instruction='''You are a helpful assistant that can answer questions about a
        document. The document is already in your context.
        Answer the user's questions based on the document.''',
)

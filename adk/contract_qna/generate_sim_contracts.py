#!/usr/bin/env
"""
This script generates a specified number of synthetic land lease agreements
using Google's Generative AI models (Gemini 2.5 Flash and Pro) through the Vertex AI SDK.

The script performs the following steps:
1.  For each agreement, it generates a unique tenant company name using a fast,
cost-effective model (Gemini 2.5 Flash).
2.  It then uses a more powerful model (Gemini 2.5 Pro) to generate a complete,
detailed land lease agreement based on a template, filling in synthetic data.
3.  Each generated agreement is converted into a PDF document.
4.  The resulting PDF is uploaded to a specified Google Cloud Storage (GCS) bucket.
5.  The local PDF file is removed after successful upload.

The script is designed to be run from the command line, with parameters for
project configuration, the number of agreements to generate, and GCS locations.
"""

import argparse
import os
import re
from google import genai
from google.cloud import storage
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
import vertexai


def generate_company_name(client: genai.Client, model: str) -> str:
    """
    Generates a single, realistic company name using a specified generative model.

    This function is designed to use a fast and cost-effective model for the
    simple task of name generation.

    Args:
        client: An initialized genai.Client instance.
        model: The name of the generative model to use (e.g., "gemini-2.5-flash").

    Returns:
        A string containing the generated company name. Returns a fallback name
        if an error occurs.
    """
    try:
        prompt = """
        Generate a single, realistic, and unique company name for a business that would lease land.
        Examples: 'Apex Logistics', 'Greenfield Innovations', 'Starlight Developments'.
        Do not add any commentary, just the company name itself.
        """
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
        )
        # Clean up the name from potential markdown or extra spaces
        return response.text.strip().replace("'", "").replace('"', '')
    except Exception as e:
        print(f"Error generating company name: {e}")
        return "Default Tenant Inc."  # Fallback name


def generate_lease_agreement(client: genai.Client, model: str, tenant_name: str) -> str:
    """
    Generates a detailed land lease agreement with synthetic data using a powerful model.

    Args:
        client: An initialized genai.Client instance.
        model: The name of the generative model to use (e.g., "gemini-2.5-pro").
        tenant_name: The name of the tenant to be included in the agreement.

    Returns:
        A string containing the full text of the generated lease agreement.
        Returns an error message if generation fails.
    """
    try:
        prompt = f"""
        Please act as a legal document assistant. Your task is to generate a complete Land Lease Agreement. Just generate the aggrement
        but do not add any extra commentary.

        **Instructions:**
        1.  The **Landlord** is **Cymbal**.
        2.  The **Tenant** is **{tenant_name}**.
        3.  Fill in all placeholders like dates, addresses, monetary values, and property descriptions with realistic, synthetic data.
        4.  The property should be a vacant lot suitable for commercial or industrial use.
        5.  The lease term should be between 5 and 15 years
        6.  The final output should be a complete, well-formatted document.
        7.  The end lease date should be after the start date

        --- START OF DOCUMENT TEMPLATE ---

        LAND LEASE AGREEMENT

        **1. PARTIES**
        This Land Lease Agreement ("Agreement") is made and entered into on [Effective Date], by and between:
        **Landlord:** Cymbal, with a principal place of business at [Landlord's Address].
        **Tenant:** {tenant_name}, with a principal place of business at
        [Tenant's Address].

        **2. PROPERTY DESCRIPTION**
        The Landlord agrees to lease to the Tenant the real property located at
        [Property Address], consisting of approximately [Number] acres, further
         described as
         [Legal Description of Property] (the "Property").

        **3. TERM OF LEASE**
        The term of this lease shall be [Number] years, commencing on
        [Start Date] and terminating on [End Date].

        **4. RENT**
        The Tenant shall pay the Landlord an annual rent of $[Amount] USD,
payable in equal monthly installments of $[Amount/12] USD.
        The first payment is due on [Start Date] and subsequent payments are
        due on the first day of each month thereafter.

        **5. USE OF PROPERTY**
        The Tenant is permitted to use the Property solely for the purpose of
        [Permitted Use, e.g., 'developing a commercial warehouse', 'operating a
         logistics hub', 'installing a solar farm']. Any other use requires the
          prior written consent of the Landlord.

        **6. IMPROVEMENTS & UTILITIES**
        Tenant may construct improvements on the property with the Landlord's
         prior written consent. All utilities, including but not limited to
         water, sewer, gas, and electricity, shall be the sole responsibility
         of the Tenant.

        **7. MAINTENANCE AND REPAIRS**
        The Tenant shall, at its own expense, maintain the Property in good and
         safe condition.

        **8. INSURANCE AND INDEMNIFICATION**
        The Tenant shall procure and maintain a commercial general liability
        insurance policy with a minimum coverage of $[Amount, e.g., '2,000,000']
        per occurrence. The Tenant agrees to indemnify and hold harmless the
        Landlord from any and all claims arising from the Tenant's use of the
        Property.

        **9. GOVERNING LAW**
        This Agreement shall be governed by and construed in accordance with
        the laws of the State of [State, e.g., 'California'].

        **10. SIGNATURES**
        IN WITNESS WHEREOF, the parties have executed this Agreement as of the
        Effective Date.

        **LANDLORD:**

        _________________________
        Cymbal
        By: [Name of Signatory], [Title]

        **TENANT:**

        _________________________
        {tenant_name}
        By: [Name of Signatory], [Title]
        """

        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                temperature=0.8,
                top_k=40,
                top_p=0.8,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                )
            ),
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt)
                    ]
                )
            ])

        return response.text
    except Exception as e:
        print(f"Error generating lease for {tenant_name}: {e}")
        return f"Could not generate lease agreement for {tenant_name}."


def simulate_agreements(client: genai.Client, number_of_agreements: int) -> list[str]:
    """
    Simulates the generation of multiple lease agreements.

    Args:
        client: An initialized genai.Client instance.
        number_of_agreements: The total number of agreements to generate.

    Returns:
        A list of strings, where each string is a generated lease agreement.
    """
    # Using a faster, more cost-effective model for the simple task
    flash_model = "gemini-2.5-flash"

    # Using a more powerful model for the complex document generation task
    pro_model = "gemini-2.5-pro"

    print(f"--- Starting Generation of {number_of_agreements} Land Lease Agreements ---")

    agreements = []

    for i in range(number_of_agreements):
        print(f"\n{'='*25} AGREEMENT {i+1} OF {number_of_agreements} {'='*25}\n")

        # Step 1: Call Gemini Flash to get a company name
        print("--> Calling Gemini 2.5 Flash to generate a tenant name...")
        tenant_name = generate_company_name(client, flash_model)
        print(f"--> Tenant Name Generated: {tenant_name}\n")

        # Step 2: Call Gemini Pro to generate the full agreement
        print(f"--> Calling Gemini 2.5 Pro to generate the lease for {tenant_name}...")
        lease_agreement = generate_lease_agreement(client, pro_model, tenant_name)

        # Step 3: Store the final result
        agreements.append(lease_agreement)

    return agreements


def create_and_upload_pdf(agreement_text: str, filename: str, bucket_name: str, gcs_path: str):
    """
    Creates a PDF file from text, preserving markdown formatting, and uploads it to GCS.

    Args:
        agreement_text: The string content of the agreement with markdown.
        filename: The local name for the PDF file to be created.
        bucket_name: The name of the GCS bucket to upload to.
        gcs_path: The destination path within the GCS bucket.
    """
    try:
        # 1. Create the document template
        doc = SimpleDocTemplate(filename, rightMargin=inch, leftMargin=inch,
                                topMargin=inch, bottomMargin=inch)

        # 2. Create a "story" (a list of flowable objects)
        story = []
        styles = getSampleStyleSheet()
        p_style = styles["Normal"]
        p_style.leading = 14  # Set space between lines

        # 3. Add content to the story, processing markdown for paragraphs
        # Split the text into paragraphs based on blank lines
        paragraphs = agreement_text.split('\n\n')

        for para_text in paragraphs:
            # Replace markdown bold (**text**) with reportlab's <b> tag
            para_text = re.sub(r'**(.*?)**', r'<b>\1</b>', para_text)

            # Replace single newlines with <br/> for line breaks within a paragraph
            para_text = para_text.replace('\n', '<br/>')

            p = Paragraph(para_text, p_style)
            story.append(p)
            # Add a spacer after each paragraph for readability
            story.append(Spacer(1, 0.2 * inch))

        # 4. Build the document
        doc.build(story)

        # 5. Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(os.path.join(gcs_path, filename))
        blob.upload_from_filename(filename)

        print(f"PDF '{filename}' uploaded to gs://{bucket_name}/{blob.name}")

    except Exception as e:
        print(f"Error creating or uploading PDF for '{filename}': {e}")
    finally:
        # 6. Clean up the local file
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed local file: {filename}")



def main():
    """
    Main function to parse arguments and orchestrate the agreement generation process.
    """
    parser = argparse.ArgumentParser(description="Generate and upload synthetic land lease agreements.")
    parser.add_argument("--project-id", required=True, help="Your Google Cloud project ID.")
    parser.add_argument("--location", required=True, help="The Google Cloud region (e.g., 'us-east4').")
    parser.add_argument("--number", type=int, default=3, help="Number of agreements to generate.")
    parser.add_argument("--bucket-name", required=True, help="GCS bucket name for uploading PDFs.")
    parser.add_argument("--gcs-path", default="capstone/land-lease-agreements",
                        help="Path within the GCS bucket to store the agreements.")

    args = parser.parse_args()

    # Initialize Vertex AI SDK
    try:
        vertexai.init(project=args.project_id, location=args.location,
                      staging_bucket=f"gs://{args.bucket_name}")
        client = genai.Client(vertexai=True, project=args.project_id, location=args.location)
    except Exception as e:
        print(f"Error initializing Vertex AI SDK: {e}")
        print("Please ensure you have authenticated with 'gcloud auth application-default login'")
        return

    # Generate the agreements
    agreements = simulate_agreements(client, args.number)

    # Create and upload PDFs
    for i, agreement in enumerate(agreements):
        filename = f"lease_agreement_{i + 1}.pdf"
        create_and_upload_pdf(agreement, filename, args.bucket_name, args.gcs_path)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()

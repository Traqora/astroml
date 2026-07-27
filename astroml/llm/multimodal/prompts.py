"""
Multimodal prompt templates for vision and document analysis tasks.

Provides templates for receipt analysis, document verification, fraud detection, etc.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


class PromptTemplate(str, Enum):
    """Predefined prompt templates."""
    RECEIPT_ANALYSIS = "receipt_analysis"
    KYC_VERIFICATION = "kyc_verification"
    FRAUD_DETECTION = "fraud_detection"
    CHART_ANALYSIS = "chart_analysis"
    DOCUMENT_SUMMARY = "document_summary"
    TEXT_EXTRACTION = "text_extraction"


@dataclass
class MultimodalPrompt:
    """A multimodal prompt with text and image components."""
    text: str
    image_paths: List[str]
    template: Optional[PromptTemplate] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "image_paths": self.image_paths,
            "template": self.template.value if self.template else None,
        }


class MultimodalPromptBuilder:
    """
    Build multimodal prompts for vision and document analysis.

    Provides templates and utilities for constructing effective prompts
    for multimodal LLM tasks.
    """

    # Template definitions
    TEMPLATES = {
        PromptTemplate.RECEIPT_ANALYSIS: """Analyze this receipt image and extract the following information:
1. Merchant name and location
2. Transaction date and time
3. Total amount
4. Items purchased (with quantities and prices if visible)
5. Payment method
6. Any loyalty points or discounts applied

Provide the response in a structured JSON format with high confidence scores.""",

        PromptTemplate.KYC_VERIFICATION: """Verify this identity document for KYC purposes. Extract:
1. Document type (passport, driver's license, ID card)
2. Full name
3. Date of birth
4. Document ID number
5. Issue and expiration dates
6. Any security features or holograms visible

Assess document quality and authenticity. Provide confidence scores for each field.""",

        PromptTemplate.FRAUD_DETECTION: """Analyze this image for fraud indicators. Identify:
1. Suspicious elements or anomalies
2. Signs of tampering or manipulation
3. Unusual patterns or inconsistencies
4. Context clues suggesting fraud
5. Overall risk level (low, medium, high, critical)

Provide reasoning for each finding.""",

        PromptTemplate.CHART_ANALYSIS: """Analyze this chart and provide:
1. Chart type and title
2. Axis labels and scale ranges
3. Data series (color and legend mapping)
4. Key values and trends
5. Anomalies or unexpected patterns
6. Insights about the data

Format values as a structured table if possible.""",

        PromptTemplate.DOCUMENT_SUMMARY: """Read and summarize this document. Provide:
1. Document type and purpose
2. Key findings or conclusions
3. Important figures or statistics
4. Action items or recommendations
5. Any tables or structured data present

Keep summary concise but comprehensive.""",

        PromptTemplate.TEXT_EXTRACTION: """Extract all visible text from this image. Preserve:
1. Original formatting (headers, paragraphs, lists)
2. Table structure if present
3. Special characters and symbols
4. Text orientation and layout

Output as clean, readable text.""",
    }

    def __init__(self):
        """Initialize prompt builder."""
        pass

    def build_receipt_analysis_prompt(
        self,
        image_path: str,
        include_loyalty: bool = True,
    ) -> MultimodalPrompt:
        """
        Build a prompt for receipt analysis.

        Args:
            image_path: Path to receipt image
            include_loyalty: Whether to extract loyalty points

        Returns:
            MultimodalPrompt object
        """
        template = self.TEMPLATES[PromptTemplate.RECEIPT_ANALYSIS]
        if include_loyalty:
            template += "\nAlso note any loyalty program information."

        return MultimodalPrompt(
            text=template,
            image_paths=[image_path],
            template=PromptTemplate.RECEIPT_ANALYSIS,
        )

    def build_kyc_verification_prompt(
        self,
        image_path: str,
        country: Optional[str] = None,
    ) -> MultimodalPrompt:
        """
        Build a prompt for KYC document verification.

        Args:
            image_path: Path to ID document image
            country: Optional country code for locale-specific validation

        Returns:
            MultimodalPrompt object
        """
        template = self.TEMPLATES[PromptTemplate.KYC_VERIFICATION]
        if country:
            template += f"\nDocument is from {country}. Apply country-specific validation rules."

        return MultimodalPrompt(
            text=template,
            image_paths=[image_path],
            template=PromptTemplate.KYC_VERIFICATION,
        )

    def build_fraud_detection_prompt(
        self,
        image_path: str,
        context: Optional[str] = None,
    ) -> MultimodalPrompt:
        """
        Build a prompt for fraud detection.

        Args:
            image_path: Path to evidence image
            context: Optional context about the case

        Returns:
            MultimodalPrompt object
        """
        template = self.TEMPLATES[PromptTemplate.FRAUD_DETECTION]
        if context:
            template = f"Context: {context}\n\n{template}"

        return MultimodalPrompt(
            text=template,
            image_paths=[image_path],
            template=PromptTemplate.FRAUD_DETECTION,
        )

    def build_chart_analysis_prompt(
        self,
        image_path: str,
        include_metrics: bool = True,
    ) -> MultimodalPrompt:
        """
        Build a prompt for chart analysis.

        Args:
            image_path: Path to chart image
            include_metrics: Whether to extract numeric metrics

        Returns:
            MultimodalPrompt object
        """
        template = self.TEMPLATES[PromptTemplate.CHART_ANALYSIS]
        if include_metrics:
            template += "\nExtract exact numeric values from the chart."

        return MultimodalPrompt(
            text=template,
            image_paths=[image_path],
            template=PromptTemplate.CHART_ANALYSIS,
        )

    def build_document_summary_prompt(
        self,
        image_paths: List[str],
        focus_areas: Optional[List[str]] = None,
    ) -> MultimodalPrompt:
        """
        Build a prompt for document summarization.

        Args:
            image_paths: Paths to document images
            focus_areas: Optional list of areas to focus on

        Returns:
            MultimodalPrompt object
        """
        template = self.TEMPLATES[PromptTemplate.DOCUMENT_SUMMARY]
        if focus_areas:
            focus_text = ", ".join(focus_areas)
            template += f"\nFocus on: {focus_text}"

        return MultimodalPrompt(
            text=template,
            image_paths=image_paths,
            template=PromptTemplate.DOCUMENT_SUMMARY,
        )

    def build_text_extraction_prompt(self, image_path: str) -> MultimodalPrompt:
        """
        Build a prompt for text extraction.

        Args:
            image_path: Path to image with text

        Returns:
            MultimodalPrompt object
        """
        return MultimodalPrompt(
            text=self.TEMPLATES[PromptTemplate.TEXT_EXTRACTION],
            image_paths=[image_path],
            template=PromptTemplate.TEXT_EXTRACTION,
        )

    def build_custom_prompt(
        self,
        text: str,
        image_paths: List[str],
    ) -> MultimodalPrompt:
        """
        Build a custom multimodal prompt.

        Args:
            text: Prompt text
            image_paths: List of image paths

        Returns:
            MultimodalPrompt object
        """
        return MultimodalPrompt(
            text=text,
            image_paths=image_paths,
            template=None,
        )

    def get_template(self, template_name: PromptTemplate) -> str:
        """
        Get a template by name.

        Args:
            template_name: Template enum value

        Returns:
            Template text
        """
        return self.TEMPLATES.get(template_name, "")

    def list_templates(self) -> List[str]:
        """
        List all available templates.

        Returns:
            List of template names
        """
        return [t.value for t in PromptTemplate]

---
tags:
  - design
  - implementation
  - bedrock_steps
  - configuration_examples
  - documentation
  - user_guide
keywords:
  - buyer seller messaging
  - shipping logistics classification
  - pydantic sub-configurations
  - category definitions
  - prompt template configuration
topics:
  - configuration examples
  - real-world use cases
  - complex classification scenarios
language: json, python
date of note: 2025-11-02
---

# Bedrock Prompt Template Generation - Buyer-Seller Messaging Classification Example

## Overview

This document provides a comprehensive real-world example of configuring the Bedrock Prompt Template Generation step for a complex buyer-seller messaging and shipping logistics classification task. This example demonstrates the full power of the Pydantic sub-configuration system with detailed category definitions, custom output formats, and sophisticated instruction sets.

## Use Case: E-commerce Dispute Classification

This example classifies buyer-seller interactions based on message content, shipping events, and delivery timing into 13 predefined categories for automated dispute resolution and customer service optimization.

## Configuration Components

### 1. SystemPromptConfig

```python
from src.cursus.steps.configs.config_bedrock_prompt_template_generation_step import SystemPromptConfig

system_prompt_config = SystemPromptConfig(
    role_definition="expert in analyzing buyer-seller messaging conversations and shipping logistics",
    expertise_areas=[
        "buyer-seller messaging analysis",
        "shipping logistics",
        "delivery timing analysis", 
        "e-commerce dispute resolution",
        "classification and categorization"
    ],
    responsibilities=[
        "classify interactions based on message content",
        "analyze shipping events and delivery timing",
        "categorize into predefined dispute categories",
        "provide evidence-based reasoning for classifications"
    ],
    behavioral_guidelines=[
        "be precise in classification decisions",
        "be objective in evidence evaluation", 
        "be thorough in timeline analysis",
        "follow exact formatting requirements",
        "consider all available evidence sources"
    ],
    tone="professional",
    include_expertise_statement=True,
    include_task_context=True
)
```

### 2. OutputFormatConfig

```python
from src.cursus.steps.configs.config_bedrock_prompt_template_generation_step import OutputFormatConfig

output_format_config = OutputFormatConfig(
    format_type="structured_text",  # Special structured text format, not JSON
    required_fields=[
        "Category",
        "Confidence Score", 
        "Key Evidence",
        "Reasoning"
    ],
    field_descriptions={
        "Category": "Exactly one category from the predefined list (case-sensitive match required)",
        "Confidence Score": "Decimal number between 0.00 and 1.00 indicating classification certainty",
        "Key Evidence": "Three subsections: Message Evidence, Shipping Evidence, Timeline Evidence - each with [sep] token separators",
        "Reasoning": "Three subsections: Primary Factors, Supporting Evidence, Contradicting Evidence - each with [sep] token separators"
    },
    validation_requirements=[
        "Category must match exactly from predefined list",
        "Confidence score must be decimal format (e.g., 0.85, not 85%)",
        "Each evidence item must start with '[sep] ' token",
        "Use exact section headers with numbers and colons",
        "No semicolons (;) anywhere in response",
        "Follow structured text format, not JSON"
    ],
    include_field_constraints=True,
    include_formatting_rules=True,
    evidence_validation_rules=[
        "Message Evidence must include direct quotes with speaker identification",
        "Shipping Evidence must include tracking events with timestamps", 
        "Timeline Evidence must show chronological sequence of events",
        "All evidence must reference specific content from input data",
        "Multiple pieces of evidence strengthen classification accuracy"
    ]
)
```

### 3. InstructionConfig

```python
from src.cursus.steps.configs.config_bedrock_prompt_template_generation_step import InstructionConfig

instruction_config = InstructionConfig(
    include_analysis_steps=True,
    include_decision_criteria=True,
    include_edge_case_handling=True,
    include_confidence_guidance=True,
    include_reasoning_requirements=True,
    step_by_step_format=True,
    include_evidence_validation=True
)
```

### 4. Category Definitions JSON

```json
[
  {
    "name": "TrueDNR",
    "description": "Delivered Not Received - Package marked as delivered but buyer claims non-receipt",
    "conditions": [
      "Package marked as delivered (EVENT_301)",
      "Buyer claims non-receipt",
      "Tracking shows delivery confirmation",
      "Buyer disputes receiving the package"
    ],
    "exceptions": [
      "Package not yet delivered",
      "Delivery attempt failed",
      "Package returned to sender"
    ],
    "key_indicators": [
      "delivered",
      "not received",
      "missing package",
      "EVENT_301",
      "delivery confirmation",
      "non-receipt"
    ],
    "examples": [
      "Hello, I have not received my package, but I see the order shows that it has been delivered, why?",
      "But I did not find any package, please refund me, thank you"
    ],
    "priority": 1,
    "validation_rules": [
      "Must have delivery confirmation event",
      "Must have buyer claim of non-receipt",
      "Timeline must show delivery before complaint"
    ]
  },
  {
    "name": "Confirmed_Delay",
    "description": "Shipment delayed due to confirmed uncontrollable external factors",
    "conditions": [
      "Delay confirmed by seller or shiptrack status",
      "External factor causing delay identified",
      "Seller acknowledges delay",
      "Shiptrack shows delay status codes"
    ],
    "exceptions": [
      "Unconfirmed delays",
      "Buyer-only claims of delay",
      "Normal transit time variations"
    ],
    "key_indicators": [
      "customs processing delays",
      "COVID-related restrictions", 
      "traffic control",
      "natural disasters",
      "weather delays",
      "war",
      "political situations",
      "labor strikes",
      "carrier facility issues",
      "confirmed delay"
    ],
    "examples": [
      "Due to customs processing delays, your package will be delayed by 5-7 business days",
      "Weather conditions have caused shipping delays in your area"
    ],
    "priority": 2,
    "validation_rules": [
      "Must have seller or system confirmation of delay",
      "Must identify specific external cause",
      "Cannot be normal transit variations"
    ]
  },
  {
    "name": "Delivery_Attempt_Failed",
    "description": "Delivery attempt unsuccessful, package returned to seller",
    "conditions": [
      "Delivery attempt unsuccessful",
      "Package returned to seller",
      "Confirmed by seller or shiptrack status",
      "Failed delivery attempt events recorded"
    ],
    "exceptions": [
      "Successful delivery",
      "Package still in transit",
      "Unconfirmed delivery issues"
    ],
    "key_indicators": [
      "failed delivery attempt",
      "return to sender",
      "delivery failure",
      "undeliverable address",
      "recipient unavailable",
      "access restrictions",
      "carrier unable to deliver"
    ],
    "examples": [
      "Your package could not be delivered due to an incorrect address and has been returned to sender",
      "Multiple delivery attempts failed - package is being returned to our facility"
    ],
    "priority": 3,
    "validation_rules": [
      "Must have confirmation of delivery failure",
      "Must show return to sender status",
      "Cannot be successful delivery"
    ]
  },
  {
    "name": "Seller_Unable_To_Ship",
    "description": "Seller offers refund directly due to shipping issues before shipment",
    "conditions": [
      "Seller offers refund without buyer request",
      "Order not shipped due to seller-side problems",
      "Seller-initiated refund before shipping",
      "Seller proactively contacts buyer about inability to ship"
    ],
    "exceptions": [
      "Buyer-requested cancellations",
      "Cases where buyer initiates cancellation",
      "Shipped items with delays"
    ],
    "key_indicators": [
      "stock unavailable",
      "out of stock",
      "shipping restrictions",
      "processing problems",
      "system issues",
      "warehouse issues",
      "fulfillment problems",
      "carrier pickup failure",
      "inventory management errors"
    ],
    "examples": [
      "We apologize, but this item is currently out of stock and we cannot fulfill your order",
      "Due to shipping restrictions to your location, we need to cancel your order and provide a full refund"
    ],
    "priority": 4,
    "validation_rules": [
      "Must occur before any shipping events",
      "Must be seller-initiated refund",
      "No shipment tracking initiated"
    ]
  },
  {
    "name": "PDA_Undeliverable",
    "description": "Item stuck in transit without status updates or seller confirmation of reason",
    "conditions": [
      "Package shows shipped/in-transit status",
      "No delivery confirmation",
      "No confirmed external delay factors",
      "Seller cannot provide specific delay/loss reason",
      "Buyer reports non-receipt"
    ],
    "exceptions": [
      "Confirmed delays",
      "Failed delivery attempts",
      "Seller unable to ship",
      "Delivered packages",
      "Packages returned to seller"
    ],
    "key_indicators": [
      "shipped",
      "in-transit",
      "no delivery confirmation",
      "no status updates",
      "stuck in transit",
      "tracking shows no delivery",
      "package missing"
    ],
    "examples": [
      "My package has been in transit for 3 weeks with no updates",
      "Tracking shows shipped but no delivery confirmation"
    ],
    "priority": 5,
    "validation_rules": [
      "Must show shipped/in-transit status",
      "Must have no delivery confirmation",
      "Must have buyer non-receipt claim"
    ]
  },
  {
    "name": "PDA_Early_Refund",
    "description": "Refund given before delivery date where product tracking later shows successful delivery",
    "conditions": [
      "Refund timestamp precedes delivery timestamp",
      "Product tracking shows successful delivery after refund",
      "No product return recorded",
      "Clear timestamp comparison between refund and delivery"
    ],
    "exceptions": [
      "Refund after delivery",
      "No delivery confirmation",
      "Return record exists"
    ],
    "key_indicators": [
      "early refund",
      "refund before delivery",
      "delivered after refund",
      "no return record",
      "timestamp verification"
    ],
    "examples": [
      "Refund processed on 2025-02-20, delivery confirmed on 2025-02-25",
      "Customer received refund but package was delivered 3 days later"
    ],
    "priority": 6,
    "validation_rules": [
      "Refund timestamp must precede delivery timestamp",
      "Delivery must be confirmed after refund",
      "No return record exists"
    ]
  },
  {
    "name": "Buyer_Received_WrongORDefective_Item",
    "description": "Product quality/condition issues with actual return expected",
    "conditions": [
      "Buyer confirms receiving item",
      "Product quality or condition issues reported",
      "Eventually becomes actual return",
      "Seller requests buyer to return the item",
      "Return shipping process initiated"
    ],
    "exceptions": [
      "Returnless refund scenarios",
      "Liquids, gels, hazardous materials",
      "Fresh items",
      "Cases where no return is expected"
    ],
    "key_indicators": [
      "damaged",
      "defective",
      "missing parts",
      "wrong size",
      "wrong color",
      "wrong model",
      "functionality problems",
      "quality issues",
      "authenticity concerns",
      "different from description"
    ],
    "examples": [
      "The item I received is damaged and doesn't work properly",
      "This is the wrong size - I ordered large but received small"
    ],
    "priority": 7,
    "validation_rules": [
      "Must include buyer confirmation of receiving item",
      "Must eventually become actual return",
      "Cannot be returnless refund eligible items"
    ]
  },
  {
    "name": "Returnless_Refund",
    "description": "Refund given without requiring customer to return the product",
    "conditions": [
      "Clear delivery confirmation or buyer does not claim non-receipt",
      "Refund given without requiring return",
      "Seller explicitly offers refund without return",
      "No return shipping label provided",
      "Product retention explicitly allowed"
    ],
    "exceptions": [
      "Return required scenarios",
      "No delivery confirmation",
      "Return shipping events follow"
    ],
    "key_indicators": [
      "liquids",
      "gels",
      "hazardous materials",
      "broken glass",
      "spilled acid",
      "fresh items",
      "broken eggs",
      "bad vegetables",
      "perishable goods",
      "keep the item",
      "no need to return",
      "returnless refund"
    ],
    "examples": [
      "This is your refund. You can keep the item.",
      "No need to return the product - here's your full refund",
      "Keep the item and here's your refund"
    ],
    "priority": 8,
    "validation_rules": [
      "Delivery confirmation must exist",
      "No return shipping events follow",
      "Explicit permission to retain product"
    ]
  },
  {
    "name": "BuyerCancellation",
    "description": "Buyer cancels order for their own reasons before delivery",
    "conditions": [
      "Buyer cancels order for personal reasons",
      "Cancellation timestamp occurs before delivery timestamp",
      "Buyer does not receive the item yet",
      "No returns involved"
    ],
    "exceptions": [
      "Post-delivery scenarios",
      "Cases with confirmed shipping delays",
      "Cases with delivery attempt failures",
      "Seller-initiated refunds"
    ],
    "key_indicators": [
      "change of mind",
      "change of plan",
      "late delivery concerns",
      "found better alternative",
      "payment issues",
      "personal circumstances change",
      "buyer cancellation"
    ],
    "examples": [
      "I changed my mind about this purchase, can I cancel?",
      "I found a better deal elsewhere, please cancel my order"
    ],
    "priority": 9,
    "validation_rules": [
      "Must occur before delivery",
      "Must be buyer-initiated",
      "No confirmed delays or delivery failures"
    ]
  },
  {
    "name": "Return_NoLongerNeeded",
    "description": "Post-delivery return initiation for good quality items no longer needed",
    "conditions": [
      "Return request timestamp occurs after delivery timestamp",
      "Buyer received the item but no longer needs it",
      "Product received is of good quality",
      "Eventually becomes actual return",
      "Buyer acknowledges receiving the item"
    ],
    "exceptions": [
      "Pre-delivery cancellations",
      "Claims of defective/damaged items",
      "Returnless refund scenarios"
    ],
    "key_indicators": [
      "changed mind",
      "no longer needed",
      "found better alternative",
      "size/fit issues",
      "duplicate purchase",
      "gift not wanted",
      "personal preference change"
    ],
    "examples": [
      "I received the item but changed my mind - can I return it?",
      "The item is fine but I no longer need it"
    ],
    "priority": 10,
    "validation_rules": [
      "Must occur after delivery confirmation",
      "Must eventually become actual return",
      "Cannot claim defective/damaged items"
    ]
  },
  {
    "name": "Product_Information_Support",
    "description": "General product information and support requests not related to refunds or returns",
    "conditions": [
      "General product information requests",
      "Documentation and information requests",
      "Product support and guidance",
      "Not related to refund or return events"
    ],
    "exceptions": [
      "Refund discussions",
      "Return discussions",
      "Shipping issues",
      "Quality complaints"
    ],
    "key_indicators": [
      "invoice copies",
      "receipts",
      "tax documents",
      "payment records",
      "product specifications",
      "warranty information",
      "how to use",
      "troubleshooting",
      "setup instructions",
      "compatibility questions"
    ],
    "examples": [
      "Can you send me a copy of my invoice for tax purposes?",
      "How do I set up this product?",
      "What are the warranty terms for this item?"
    ],
    "priority": 11,
    "validation_rules": [
      "Must be information request only",
      "Cannot involve refund or return discussion",
      "Must be product or service related"
    ]
  },
  {
    "name": "Insufficient_Information",
    "description": "Ultimate fallback category when context is missing or insufficient",
    "conditions": [
      "Information from dialogue and/or ship track events insufficient",
      "Lack of one or both input data sources",
      "Message cut off or incomplete dialogue",
      "Available data insufficient for other categories"
    ],
    "exceptions": [
      "Clear classification possible with available data",
      "Sufficient information for other categories"
    ],
    "key_indicators": [
      "incomplete dialogue",
      "missing ship track",
      "corrupted messages",
      "unreadable content",
      "insufficient data",
      "unclear context"
    ],
    "examples": [
      "[Message cut off or incomplete]",
      "[No ship track data available]",
      "[Corrupted or unreadable messages]"
    ],
    "priority": 12,
    "validation_rules": [
      "Use only when no other category fits",
      "Must indicate specific data insufficiency",
      "Default category for unclear cases"
    ]
  }
]
```

## Complete Configuration Example

### Full BedrockPromptTemplateGenerationConfig Setup

```python
from src.cursus.steps.configs.config_bedrock_prompt_template_generation_step import (
    BedrockPromptTemplateGenerationConfig,
    SystemPromptConfig,
    OutputFormatConfig,
    InstructionConfig
)

# Create the complete configuration
config = BedrockPromptTemplateGenerationConfig(
    # Basic template settings
    template_task_type="buyer_seller_classification",
    template_style="structured",
    validation_level="comprehensive",
    template_version="2.0",
    
    # Input configuration
    input_placeholders=["dialogue", "shiptrack", "max_estimated_arrival_date"],
    
    # Output configuration
    output_format_type="structured_text",
    required_output_fields=["Category", "Confidence Score", "Key Evidence", "Reasoning"],
    
    # Template features
    include_examples=True,
    generate_validation_schema=True,
    
    # Pydantic sub-configurations
    system_prompt_settings=SystemPromptConfig(
        role_definition="expert in analyzing buyer-seller messaging conversations and shipping logistics",
        expertise_areas=[
            "buyer-seller messaging analysis",
            "shipping logistics",
            "delivery timing analysis", 
            "e-commerce dispute resolution",
            "classification and categorization"
        ],
        responsibilities=[
            "classify interactions based on message content",
            "analyze shipping events and delivery timing",
            "categorize into predefined dispute categories",
            "provide evidence-based reasoning for classifications"
        ],
        behavioral_guidelines=[
            "be precise in classification decisions",
            "be objective in evidence evaluation", 
            "be thorough in timeline analysis",
            "follow exact formatting requirements",
            "consider all available evidence sources"
        ],
        tone="professional",
        include_expertise_statement=True,
        include_task_context=True
    ),
    
    output_format_settings=OutputFormatConfig(
        format_type="structured_text",
        required_fields=["Category", "Confidence Score", "Key Evidence", "Reasoning"],
        field_descriptions={
            "Category": "Exactly one category from the predefined list (case-sensitive match required)",
            "Confidence Score": "Decimal number between 0.00 and 1.00 indicating classification certainty",
            "Key Evidence": "Three subsections: Message Evidence, Shipping Evidence, Timeline Evidence - each with [sep] token separators",
            "Reasoning": "Three subsections: Primary Factors, Supporting Evidence, Contradicting Evidence - each with [sep] token separators"
        },
        validation_requirements=[
            "Category must match exactly from predefined list",
            "Confidence score must be decimal format (e.g., 0.85, not 85%)",
            "Each evidence item must start with '[sep] ' token",
            "Use exact section headers with numbers and colons",
            "No semicolons (;) anywhere in response",
            "Follow structured text format, not JSON"
        ],
        include_field_constraints=True,
        include_formatting_rules=True,
        evidence_validation_rules=[
            "Message Evidence must include direct quotes with speaker identification",
            "Shipping Evidence must include tracking events with timestamps", 
            "Timeline Evidence must show chronological sequence of events",
            "All evidence must reference specific content from input data",
            "Multiple pieces of evidence strengthen classification accuracy"
        ]
    ),
    
    instruction_settings=InstructionConfig(
        include_analysis_steps=True,
        include_decision_criteria=True,
        include_edge_case_handling=True,
        include_confidence_guidance=True,
        include_reasoning_requirements=True,
        step_by_step_format=True,
        include_evidence_validation=True
    )
)
```

## Generated Template Structure

### System Prompt (Generated)
```
You are an expert in analyzing buyer-seller messaging conversations and shipping logistics with extensive knowledge in buyer-seller messaging analysis, shipping logistics, delivery timing analysis, e-commerce dispute resolution, classification and categorization. Your task is to classify interactions based on message content, analyze shipping events and delivery timing, categorize into predefined dispute categories, provide evidence-based reasoning for classifications. Always be precise in classification decisions, be objective in evidence evaluation, be thorough in timeline analysis, follow exact formatting requirements, consider all available evidence sources in your analysis.
```

### User Prompt Template (Generated)
```
Categories and their criteria:

1. TrueDNR
    - Delivered Not Received - Package marked as delivered but buyer claims non-receipt
    - Key elements:
        * delivered
        * not received
        * missing package
        * EVENT_301
        * delivery confirmation
        * non-receipt
    - Conditions:
        * Package marked as delivered (EVENT_301)
        * Buyer claims non-receipt
        * Tracking shows delivery confirmation
        * Buyer disputes receiving the package
    - Must NOT include:
        * Package not yet delivered
        * Delivery attempt failed
        * Package returned to sender

[... continues with all 12 categories ...]

Analysis Instructions:

Please analyze:
Dialogue: {dialogue}
Shiptrack: {shiptrack}
Max_estimated_arrival_date: {max_estimated_arrival_date}

Provide your analysis in the following structured format:

1. Carefully review all provided data
2. Identify key patterns and indicators
3. Match against category criteria
4. Select the most appropriate category
5. Validate evidence against conditions and exceptions
6. Provide confidence assessment and reasoning

Decision Criteria:
- Base decisions on explicit evidence in the data
- Consider all category conditions and exceptions
- Choose the category with the strongest evidence match
- Provide clear reasoning for your classification

Key Evidence Validation:
- Evidence MUST align with at least one condition for the selected category
- Evidence MUST NOT match any exceptions listed for the selected category
- Evidence should reference specific content from the input data
- Multiple pieces of supporting evidence strengthen the classification

## Required Output Format

**CRITICAL: Follow this exact format for automated parsing**

```
1. Category: [EXACT_CATEGORY_NAME]

2. Confidence Score: [0.XX]

3. Key Evidence:
   * Message Evidence:
     [sep] [Evidence item 1]
     [sep] [Evidence item 2]
   * Shipping Evidence:
     [sep] [Evidence item 1]
     [sep] [Evidence item 2]
   * Timeline Evidence:
     [sep] [Evidence item 1]
     [sep] [Evidence item 2]

4. Reasoning:
   * Primary Factors:
     [sep] [Factor 1]
     [sep] [Factor 2]
   * Supporting Evidence:
     [sep] [Supporting item 1]
     [sep] [Supporting item 2]
   * Contradicting Evidence:
     [sep] [Contradicting item 1 OR "None"]
```

**Formatting Rules:**
- Use exact section headers with numbers and colons
- Category name must match exactly from provided list
- Confidence score as decimal (e.g., 0.85, not 85% or "high")
- Each evidence/reasoning item starts with "[sep] " (including space)
- Use asterisk (*) for subsection headers with exact spacing
- No semicolons (;) anywhere in response
- No additional formatting or markdown

Do not include any text before or after the structured format. Only return the structured analysis.
```

## Usage Examples

### Step Builder Integration

```python
from src.cursus.steps.builders.builder_bedrock_prompt_template_generation_step import BedrockPromptTemplateGenerationStepBuilder

# Create step builder with configuration
builder = BedrockPromptTemplateGenerationStepBuilder(
    config=config,
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Create the step
step = builder.create_step(
    inputs={
        "category_definitions": category_definitions_s3_path
    },
    outputs={
        "prompt_templates": f"{base_output_path}/templates",
        "template_metadata": f"{base_output_path}/metadata", 
        "validation_schema": f"{base_output_path}/schema"
    }
)
```

### Pipeline Integration

```python
from sagemaker.workflow import Pipeline

# Create pipeline with the step
pipeline = Pipeline(
    name="buyer-seller-classification-pipeline",
    steps=[step],
    sagemaker_session=sagemaker_session
)

# Execute pipeline
execution = pipeline.start()
```

## Key Features Demonstrated

### 1. Complex Category System
- **13 distinct categories** with detailed conditions and exceptions
- **Priority-based classification** with tier hierarchy
- **Comprehensive validation rules** for each category
- **Real-world examples** for each classification type

### 2. Advanced Output Format
- **Structured text format** (not JSON) with specific token requirements
- **Multi-section evidence** (Message, Shipping, Timeline)
- **Detailed reasoning structure** (Primary, Supporting, Contradicting)
- **Strict formatting rules** for automated parsing

### 3. Sophisticated Instructions
- **Step-by-step analysis process** with clear decision criteria
- **Evidence validation requirements** with specific rules
- **Edge case handling** for missing data scenarios
- **Confidence scoring guidance** based on data completeness

### 4. Type-Safe Configuration
- **Pydantic sub-configurations** with full validation
- **IDE autocompletion** for all configuration options
- **Comprehensive defaults** with easy customization
- **Error prevention** through compile-time validation

## Benefits of This Approach

### 1. Maintainability
- **Centralized configuration** in typed Pydantic models
- **Clear separation** of concerns (system, output, instructions)
- **Version control friendly** with structured definitions
- **Easy updates** without touching template generation code

### 2. Reliability
- **Type safety** prevents configuration errors
- **Validation** ensures all required fields are present
- **Consistent output** through structured formatting
- **Error handling** with graceful degradation

### 3. Scalability
- **Reusable components** across different classification tasks
- **Extensible design** for adding new categories or fields
- **Template inheritance** for similar use cases
- **Performance optimization** through caching

### 4. Developer Experience
- **IDE support** with full autocompletion
- **Self-documenting** configuration with field descriptions
- **Easy testing** with typed configuration objects
- **Clear examples** for rapid implementation

This comprehensive example demonstrates how the Pydantic sub-configuration system enables sophisticated, maintainable, and reliable prompt template generation for complex real-world classification tasks.
</content>

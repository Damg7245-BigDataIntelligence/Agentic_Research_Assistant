import os
import tiktoken
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv

load_dotenv()

def generate_response_with_gemini(query, context=None, model_name="gemini-1.5-pro", response_type="default"):
    """
    Generate a response using Google Gemini model
    response_type: 'default' or 'web_analysis' to handle different prompt structures
    """
    try:
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: Google API key not configured. Please set the GOOGLE_API_KEY environment variable.", None
        
        configure(api_key=api_key)
        model = GenerativeModel(model_name)
        
        # For token counting
        encoder = tiktoken.get_encoding("cl100k_base")
        
        if response_type == "web_analysis":
            prompt = f"""
            You are a financial analyst specializing in NVIDIA and the tech industry.
            Analyze the following recent news and trends about NVIDIA to provide strategic insights.

            {context}

            Please provide a structured analysis with the following sections:

            1. KEY DEVELOPMENTS:
            - List the most significant recent events or announcements
            - Highlight their importance in the industry context

            2. MARKET IMPACT:
            - Analyze potential effects on NVIDIA's market position
            - Discuss competitive implications
            - Identify any market opportunities or challenges

            3. INDUSTRY TRENDS:
            - Identify broader patterns in the semiconductor/AI industry
            - Connect these trends to NVIDIA's strategy
            - Note any emerging market dynamics

            4. FUTURE OUTLOOK:
            - Provide forward-looking analysis
            - Highlight potential opportunities and risks
            - Suggest areas to watch

            Format your response in clear sections with bullet points for easy reading.
            Focus on factual analysis based on the provided information.
            """
        elif context:
            # Original RAG prompt
            prompt = f"""
            Please answer the following question based only on the provided context. 
            If the context doesn't contain the information needed to answer the question, say that you don't have enough information.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
        else:
            prompt = query
        
        # Count input tokens before API call
        input_tokens = len(encoder.encode(prompt))
        
        # Generate response
        response = model.generate_content(prompt)
        answer_text = response.text
        
        # Count output tokens
        output_tokens = len(encoder.encode(answer_text))
        
        # Calculate costs based on pricing for Gemini models
        input_cost = (input_tokens / 1000) * 0.0000125
        output_cost = (output_tokens / 1000) * 0.00005
        total_cost = input_cost + output_cost
        
        token_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model_name
        }
        
        return answer_text, token_info
    except Exception as e:
        return f"Error generating response: {str(e)}", None
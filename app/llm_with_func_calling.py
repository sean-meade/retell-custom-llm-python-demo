import datetime

from openai import AsyncOpenAI
import os
import json
import httpx
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List

# Initial greeting message sent at the start of a conversation
begin_sentence = "Hi there, you're through to Intercom Technical Support. How can I help today?"

# System prompt that defines the AI agent's role, capabilities, and behavior
# This is combined with additional instructions in prepare_prompt()
agent_prompt = "You are an Intercom Technical Support Specialist (TSS). You provide expert-level, technically support for Intercom’s Customer Service Suite, including Fin, Inbox, Workflows, APIs, Webhooks, Messenger, and integrations. You own issues from first contact through resolution. You identify the core Problem to be Solved, troubleshoot methodically, and communicate clearly, consisely and confidently. You understand REST APIs, authentication, JSON payloads, rate limits, and SaaS architecture. You never fabricate features or speculate without labeling uncertainty. When appropriate, you escalate with structured summaries including reproduction steps and relevant technical context. You optimize for brevity, accuracy, clarity, efficiency, and customer confidence. Tone is friendly, professional, human, concise, and structured. Avoid filler. Avoid repetition. Never expose internal reasoning. Today’s date is {}.".format(datetime.date.today().strftime('%A, %B %d, %Y'))


class LlmClient:
    """
    Client for interacting with OpenAI's API to handle LLM conversations
    with function calling capabilities for call control (e.g., ending calls).
    """
    
    def __init__(self):
        """
        Initialize the OpenAI client using organization ID and API key from environment variables.
        """
        self.client = AsyncOpenAI(
            organization=os.environ["OPENAI_ORGANIZATION_ID"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def draft_begin_message(self):
        """
        Creates the initial greeting message that starts the conversation.
        
        Returns:
            ResponseResponse: A response object containing the greeting message
        """
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        """
        Converts the conversation transcript (list of Utterance objects) into
        the format expected by OpenAI's API (list of message dictionaries).
        
        Args:
            transcript: List of Utterance objects from the conversation history
            
        Returns:
            List of message dictionaries with "role" and "content" keys
        """
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                # Agent utterances become "assistant" role messages
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                # User utterances become "user" role messages
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        """
        Constructs the complete prompt/message list for the OpenAI API call.
        Includes system instructions, conversation history, and optional reminder context.
        
        Args:
            request: Request object containing transcript and interaction metadata
            
        Returns:
            List of message dictionaries ready to send to OpenAI API
        """
        # Start with system message containing role definition and instructions
        prompt = [
            {
                "role": "system",
                "content": "## Objective\nYou are Dan acting as a Technical Support Specialist in a live support interaction. Deliver precise, AI-native, high-trust technical support.\n\n## Communication Rules\n- Be concise and structured.\n- never list out a URL if there is a URL involved give them a general description of whats involved and where to find it\n- Clarify before assuming.\n- Use simple language unless technical depth is required.\n- Break complex answers into steps.\n- Avoid repetition.\n- Never invent features.\n\n## Internal Troubleshooting Logic\n- Identify feature or API involved.\n- Determine expected behavior vs defect.\n- Validate configuration and permissions.\n- Suggest logs or payload checks when relevant.\n- Provide actionable next steps.\n\n## Escalation\nEscalate confirmed product defects or engineering-level issues with summarized findings.\n\n## Response Pattern\nAcknowledge → Align on issue → Explain cause → Provide steps → Offer next action.\n\nOperate with high ownership and technical confidence. End the call saying thanks for reaching out today use todays date." + agent_prompt,
            }
        ]
        # Add conversation history (converted from transcript format)
        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        for message in transcript_messages:
            prompt.append(message)

        # If this is a reminder scenario (user hasn't responded), add context
        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    def prepare_functions(self):
        """
        Defines the function calling schema for OpenAI's tool calling feature.
        This tells the LLM what functions are available and when/how to call them.
        
        Returns:
            List of function definition dictionaries in OpenAI's tool calling format
            
        Note: The returned format matches OpenAI's tools API specification:
        - For newer API versions, the format with "type": "function" is correct
        - Each function needs: name, description, and parameters (JSON schema)
        """
        functions = [
            {
                "type": "function",  # Required for tools API format
                "function": {
                    "name": "end_call",  # Function name the LLM will reference
                    # Improved description: clearer criteria for when to end the call
                    "description": "End the call when the user explicitly requests to end the conversation, says goodbye, or indicates they are done. Use this function to gracefully conclude the support session.",
                    "parameters": {
                        "type": "object",  # JSON schema type
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The closing message you will say before ending the call with the customer. Should thank them and mention today's date.",
                            },
                        },
                        "required": ["message"],  # Required parameters
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_account_status",
                    "description": "Check if a user's account is active by making an API request. Use this when the user asks about their account status, whether their account is active, or if they need to verify their account status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_identifier": {
                                "type": "string",
                                "description": "The user identifier to check (phone number, email address, or account ID). Extract this from the conversation or ask the user if not provided.",
                            },
                        },
                        "required": ["user_identifier"],
                    },
                },
            },
        ]
        return functions

    async def draft_response(self, request: ResponseRequiredRequest):
        """
        Main method that generates LLM responses with function calling support.
        Streams responses and handles tool calls (function invocations) from the LLM.
        
        Args:
            request: Request object containing transcript and metadata
            
        Yields:
            ResponseResponse objects containing either:
            - Streaming text content chunks
            - Function call results (e.g., end_call)
            - Final completion signal
        """
        # Build the prompt with system message and conversation history
        prompt = self.prepare_prompt(request)
        
        # Track function calls during streaming
        # func_call: Dict to store function call metadata (id, name, arguments)
        # func_arguments: String accumulator for JSON arguments (streamed incrementally)
        func_call = {}
        func_arguments = ""
        
        # Make streaming API call with function definitions
        stream = await self.client.chat.completions.create(
            # model="gpt-4-turbo-preview",  # Or use a 3.5 model for speed
            model="gpt-4o-mini",
            messages=prompt,
            stream=True,  # Enable streaming for real-time responses
            # Pass function definitions so LLM knows what functions are available
            tools=self.prepare_functions(),
        )

        # Process streaming chunks from the API
        async for chunk in stream:
            # Skip empty chunks
            if len(chunk.choices) == 0:
                continue
            
            choice = chunk.choices[0]
            
            # Check if this chunk contains a tool call (function invocation)
            if choice.delta.tool_calls:
                # Handle multiple tool calls - iterate through all tool calls in this chunk
                for tool_call_delta in choice.delta.tool_calls:
                    # If tool_call_delta.id exists, this is the start of a new function call
                    if tool_call_delta.id:
                        # If we already have a func_call with a different ID, we've moved to a new call
                        # In this case, we should finish processing the previous call
                        # For now, we'll handle only the first tool call (most common case)
                        if func_call and func_call.get("id") != tool_call_delta.id:
                            # New function call started - break to process the previous one
                            break
                        
                        # Initialize or update function call metadata
                        if not func_call:
                            func_call = {
                                "id": tool_call_delta.id,  # Unique ID for this tool call
                                "func_name": tool_call_delta.function.name or "",  # Function name
                                "arguments": {},  # Will be populated after parsing JSON
                            }
                            # Debug logging: function call detected
                            print(f"DEBUG: Function call detected - name: {func_call['func_name']}, id: {func_call['id']}")
                    
                    # Accumulate function arguments (JSON string fragments arrive incrementally)
                    if tool_call_delta.function.arguments:
                        func_arguments += tool_call_delta.function.arguments
                        # Debug logging: show partial arguments as they arrive
                        print(f"DEBUG: Accumulating function arguments (partial): {tool_call_delta.function.arguments[:50]}...")

            # Check if this chunk contains regular text content (not a function call)
            if choice.delta.content:
                # Yield streaming text content as it arrives
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=choice.delta.content,  # Partial text chunk
                    content_complete=False,  # More content coming
                    end_call=False,
                )
                yield response

        # After streaming completes, handle any function calls that were made
        if func_call:
            # Debug logging: function call detected after streaming
            print(f"DEBUG: Processing function call - name: {func_call.get('func_name')}, accumulated arguments length: {len(func_arguments)}")
            # A function was called - execute it
            if func_call["func_name"] == "end_call":
                print(f"DEBUG: end_call function invoked. Raw arguments string: {repr(func_arguments)}")
                # Parse the accumulated JSON arguments string into a dictionary
                # Add error handling for malformed or incomplete JSON
                try:
                    if not func_arguments.strip():
                        # No arguments provided - use a default message
                        func_call["arguments"] = {"message": "Thank you for reaching out today. Have a great day!"}
                    else:
                        func_call["arguments"] = json.loads(func_arguments)
                    
                    # Validate that required "message" parameter exists
                    if "message" not in func_call["arguments"]:
                        # Missing required parameter - use default message
                        func_call["arguments"]["message"] = "Thank you for reaching out today. Have a great day!"
                    
                    # Create response with the end_call message and set end_call flag
                    print(f"DEBUG: end_call function executed successfully. Message: {func_call['arguments']['message']}")
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=func_call["arguments"]["message"],  # Message to say before ending
                        content_complete=True,  # This is the final response
                        end_call=True,  # Signal to end the call
                    )
                    yield response
                except json.JSONDecodeError as e:
                    # JSON parsing failed - log error and continue without ending call
                    # In production, you might want to log this to a monitoring system
                    print(f"Error parsing function arguments JSON: {e}. Arguments string: {func_arguments}")
                    # Fall through to normal completion without ending call
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content="",  # No additional content
                        content_complete=True,  # Streaming finished
                        end_call=False,  # Continue the conversation (don't end on error)
                    )
                    yield response
            elif func_call["func_name"] == "check_account_status":
                print(f"DEBUG: check_account_status function invoked. Raw arguments string: {repr(func_arguments)}")
                try:
                    # Parse the accumulated JSON arguments string into a dictionary
                    if not func_arguments.strip():
                        func_call["arguments"] = {}
                    else:
                        func_call["arguments"] = json.loads(func_arguments)
                    
                    # Validate that required "user_identifier" parameter exists
                    if "user_identifier" not in func_call["arguments"]:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content="I need a user identifier (phone number, email, or account ID) to check the account status. Could you please provide that?",
                            content_complete=True,
                            end_call=False,
                        )
                        yield response
                        return
                    
                    user_identifier = func_call["arguments"]["user_identifier"]
                    
                    # Get API endpoint from environment variable (with fallback)
                    
                    if os.environ.get("DEV", False):
                        api_endpoint = os.environ.get("ACCOUNT_STATUS_API_ENDPOINT_DEV", "https://dev.planmyrunapi.onrender.com/api/account/status")
                    else:
                        api_endpoint = os.environ.get("ACCOUNT_STATUS_API_ENDPOINT", "https://planmyrunapi.onrender.com/api/account/status")
                        
                    if not api_endpoint:
                        # If no API endpoint configured, return a message indicating this
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content="I'm unable to check account status right now as the API endpoint is not configured. Please contact support directly.",
                            content_complete=True,
                            end_call=False,
                        )
                        yield response
                        return
                    
                    # Make API request to check account status
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            # You can customize the request format based on your API requirements
                            # Example: POST request with JSON body
                            headers = {"Content-Type": "application/json"}
                            # Add authorization header if needed
                            if os.environ.get('DEV') != "True":
                                headers["Authorization"] = f"Bearer {os.environ.get('ACCOUNT_STATUS_API_KEY', '')}"
                            
                            api_response = await client.post(
                                api_endpoint,
                                json={"user_identifier": user_identifier},
                                headers=headers,
                            )
                            api_response.raise_for_status()  # Raise exception for HTTP errors
                            account_data = api_response.json()
                            
                            # Format the response based on API response structure
                            # Adjust this based on your actual API response format
                            if account_data.get("active", False) or account_data.get("status") == "active":
                                status_message = f"Good news! The account for {user_identifier} is active and in good standing."
                            else:
                                status_message = f"I've checked the account for {user_identifier}. The account status is: {account_data.get('status', 'inactive')}. {account_data.get('message', '')}"
                            
                            response = ResponseResponse(
                                response_id=request.response_id,
                                content=status_message,
                                content_complete=True,
                                end_call=False,
                            )
                            yield response
                    except httpx.HTTPStatusError as e:
                        # Handle HTTP errors (4xx, 5xx)
                        error_message = f"I encountered an error checking the account status. The service returned status {e.response.status_code}."
                        if e.response.status_code == 404:
                            error_message = f"I couldn't find an account associated with {user_identifier}. Could you verify the identifier?"
                        elif e.response.status_code == 401:
                            error_message = "I'm unable to check account status due to authentication issues. Please contact support directly."
                        
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content=error_message,
                            content_complete=True,
                            end_call=False,
                        )
                        yield response
                    except httpx.TimeoutException:
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content="The account status check is taking longer than expected. Please try again in a moment or contact support directly.",
                            content_complete=True,
                            end_call=False,
                        )
                        yield response
                    except Exception as e:
                        print(f"Error making account status API request: {e}")
                        response = ResponseResponse(
                            response_id=request.response_id,
                            content="I encountered an error while checking the account status. Please try again or contact support directly.",
                            content_complete=True,
                            end_call=False,
                        )
                        yield response
                except json.JSONDecodeError as e:
                    print(f"Error parsing function arguments JSON: {e}. Arguments string: {func_arguments}")
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content="I encountered an error processing your request. Could you please provide the user identifier again?",
                        content_complete=True,
                        end_call=False,
                    )
                    yield response
            else:
                # Unknown function name - log and continue
                print(f"Unknown function called: {func_call.get('func_name')}")
                # Fall through to normal completion
                response = ResponseResponse(
                    response_id=request.response_id,
                    content="",  # No additional content
                    content_complete=True,  # Streaming finished
                    end_call=False,  # Continue the conversation
                )
                yield response
        else:
            # No function was called - just signal that streaming is complete
            print(f"DEBUG: No function call detected. func_call: {func_call}, func_arguments length: {len(func_arguments)}")
            response = ResponseResponse(
                response_id=request.response_id,
                content="",  # No additional content
                content_complete=True,  # Streaming finished
                end_call=False,  # Continue the conversation
            )
            yield response

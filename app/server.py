import json
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from retell import Retell
from .custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from .llm_with_func_calling import LlmClient  # or use .llm_with_func_calling

# Load environment variables from .env file (overrides existing env vars)
# load_dotenv(override=True)

# Initialize FastAPI application instance
app = FastAPI()

# Initialize Retell API client using API key from environment variables
# Used for webhook signature verification
retell = Retell(api_key=os.environ["RETELL_API_KEY"])


@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle webhook POST requests from Retell server.
    
    This endpoint receives events from Retell server including:
    - call_started: When a new call begins
    - call_ended: When a call terminates
    - call_analyzed: When call analysis is complete
    
    The webhook verifies the request signature to ensure it's from Retell.
    
    Args:
        request: FastAPI Request object containing webhook payload
        
    Returns:
        JSONResponse: 200 if successful, 401 if unauthorized, 500 on error
    """
    try:
        # Parse JSON payload from webhook request
        post_data = await request.json()
        
        # Verify webhook signature to ensure request is from Retell
        # Uses X-Retell-Signature header and API key for verification
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=str(os.environ["RETELL_API_KEY"]),
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        
        # Reject unauthorized requests
        if not valid_signature:
            print(
                "Received Unauthorized",
                post_data["event"],
                post_data["data"]["call_id"],
            )
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
        
        # Handle different event types
        if post_data["event"] == "call_started":
            print("Call started event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_ended":
            print("Call ended event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_analyzed":
            print("Call analyzed event", post_data["data"]["call_id"])
        else:
            print("Unknown event", post_data["event"])
        
        # Acknowledge receipt of webhook
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as err:
        # Handle any errors during webhook processing
        print(f"Error in webhook: {err}")
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )


@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    """
    WebSocket handler for real-time communication with Retell server.
    
    This endpoint establishes a bidirectional WebSocket connection with Retell server
    to exchange text input/output. The Retell server sends transcriptions and other
    information, and this server generates LLM responses and sends them back.
    
    Args:
        websocket: WebSocket connection instance
        call_id: Unique identifier for the call (extracted from URL path)
        
    Flow:
        1. Accept WebSocket connection
        2. Send configuration to Retell server
        3. Send initial greeting message
        4. Process incoming messages and generate responses
        5. Handle disconnections and errors
    """
    try:
        # Accept the WebSocket connection from Retell server
        await websocket.accept()
        
        # Initialize LLM client for generating responses
        llm_client = LlmClient()

        # Send optional configuration to Retell server
        # auto_reconnect: Allow automatic reconnection if connection drops
        # call_details: Request call metadata to be sent
        config = ConfigResponse(
            response_type="config",
            config={
                "auto_reconnect": True,  # Enable automatic reconnection
                "call_details": True,  # Request call details in messages
            },
            response_id=1,
        )
        await websocket.send_json(config.__dict__)

        # Send first message (greeting) to signal server is ready
        # This is the initial message that starts the conversation
        response_id = 0
        first_event = llm_client.draft_begin_message()
        await websocket.send_json(first_event.__dict__)

        async def handle_message(request_json):
            """
            Process incoming messages from Retell server.
            
            There are 5 types of interaction_type:
            - call_details: Call metadata (logged but not processed)
            - ping_pong: Keepalive messages (responded to immediately)
            - update_only: Transcript updates (no response needed)
            - response_required: User spoke, LLM response needed
            - reminder_required: User hasn't responded, reminder needed
            
            Only response_required and reminder_required trigger LLM responses.
            
            Args:
                request_json: Dictionary containing message data from Retell server
            """
            nonlocal response_id  # Allow modifying outer scope variable

            # Handle call_details: Log call metadata but don't generate response
            if request_json["interaction_type"] == "call_details":
                print(json.dumps(request_json, indent=2))
                return
            
            # Handle ping_pong: Respond immediately to keep connection alive
            if request_json["interaction_type"] == "ping_pong":
                await websocket.send_json(
                    {
                        "response_type": "ping_pong",
                        "timestamp": request_json["timestamp"],  # Echo back timestamp
                    }
                )
                return
            
            # Handle update_only: Transcript updates, no response needed
            if request_json["interaction_type"] == "update_only":
                return
            
            # Handle response_required and reminder_required: Generate LLM response
            if (
                request_json["interaction_type"] == "response_required"
                or request_json["interaction_type"] == "reminder_required"
            ):
                # Update response_id to track the latest request
                response_id = request_json["response_id"]
                
                # Create request object from incoming message
                request = ResponseRequiredRequest(
                    interaction_type=request_json["interaction_type"],
                    response_id=response_id,
                    transcript=request_json["transcript"],  # Conversation history
                )
                
                # Log the request for debugging
                print(
                    f"""Received interaction_type={request_json['interaction_type']}, response_id={response_id}, last_transcript={request_json['transcript'][-1]['content']}"""
                )

                # Generate and stream LLM responses
                # draft_response() is an async generator that yields ResponseResponse objects
                async for event in llm_client.draft_response(request):
                    # Send each response chunk/event back to Retell server
                    await websocket.send_json(event.__dict__)
                    
                    # If a new response is needed (response_id changed), abandon current response
                    # This prevents sending stale responses when user interrupts
                    if request.response_id < response_id:
                        break  # new response needed, abandon this one

        # Main message loop: continuously receive and process messages from Retell server
        # Each message is handled in a separate task to allow concurrent processing
        async for data in websocket.iter_json():
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        # Client (Retell server) closed the connection normally
        print(f"LLM WebSocket disconnected for {call_id}")
    except ConnectionTimeoutError as e:
        # Connection timed out
        print(f"Connection timeout error for {call_id}")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Error in LLM WebSocket: {e} for {call_id}")
        # Close connection with error code 1011 (internal error)
        await websocket.close(1011, "Server error")
    finally:
        # Always log when connection closes (cleanup)
        print(f"LLM WebSocket connection closed for {call_id}")

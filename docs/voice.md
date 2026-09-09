# WAC AI Assistant --- Gemini Live Voice

## Purpose

The voice subsystem provides real-time voice interaction through Gemini
Live.

It is separate from normal text chat because it uses a real-time audio
session.

## Architecture

``` text
User Microphone
      |
React Voice UI
      |
/voice/token
      |
FastAPI
      |
Gemini authentication/token creation
      |
Gemini Live Model
      |
Audio Input / Audio Output
      |
React Voice UI
```

## Live Model

The Live model is configured independently with:

``` text
GEMINI_LIVE_MODEL
```

The normal text model setting `GEMINI_MODEL` does not select the Live
model.

Therefore the text-chat dropdown does not need to contain the Live
model.

## Ephemeral Token

`GET /voice/token` creates a short-lived token for a Live session.

The token configuration includes:

-   Dedicated Live model
-   Audio response modality
-   Input audio transcription
-   Output audio transcription
-   WAC-specific system instruction

The frontend uses the token to establish the Live session.

## WAC-Only Voice Behavior

The Live system instruction defines the assistant as a WAC-specific
assistant.

WAC-related questions should be answered using available WAC knowledge.
Unrelated questions should receive the controlled WAC-only refusal. The
model should not invent WAC information.

## Voice Interruption

Real-time audio must handle user interruption.

The frontend tracks active audio sources so that an existing assistant
response can be stopped when the user starts speaking.

``` text
Assistant audio playing
        |
User interrupts
        |
Stop active audio sources
        |
Process new input
        |
Play new response
```

This prevents an older response from continuing after an interruption.

## Voice Persistence

`POST /voice/message` stores user and assistant transcript messages in
the conversation system and records voice usage.

## Voice Analytics

The voice request can include:

-   Input audio duration
-   Output audio duration
-   Input tokens
-   Output tokens
-   Latency
-   Live session ID

These values are passed to `UsageService`.

## Text vs Live

``` text
Text:
Frontend -> /chat -> AIService -> Gemini/Groq

Voice:
Frontend -> /voice/token -> Gemini Live
Frontend -> /voice/message -> Persistence + Analytics
```

## Function Calling in Live

If Live tool/function calling is configured, Live can request an
application capability through a function-call event. The application
must execute only approved tools and return the tool result to the Live
session.

## Presentation Explanation

The voice system uses Gemini Live as a separate real-time audio path.
The backend creates a short-lived Live authentication token configured
with the dedicated Live model and WAC system instruction. The frontend
establishes the audio session and explicitly stops active audio sources
when the user interrupts.

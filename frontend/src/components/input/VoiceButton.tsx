import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  GeminiLiveService,
  type VoiceState,
} from "../../services/geminilive";

import {
  useChat,
} from "../../context/ChatContext";


export default function VoiceButton() {
  // ==========================================================
  // CHAT CONTEXT
  // ==========================================================

  const {
    conversationId,

    addVoiceUserMessage,

    addVoiceAssistantMessage,

    updateStreamingAssistantMessage,

    setVoiceConversationId,
  } = useChat();


  // ==========================================================
  // SERVICE
  // ==========================================================

  const service =
    useRef<GeminiLiveService | null>(
      null
    );


  // ==========================================================
  // STATE
  // ==========================================================

  const [
    state,
    setState,
  ] = useState<VoiceState>(
    "idle"
  );


  const [
    error,
    setError,
  ] = useState<string | null>(
    null
  );


  // ==========================================================
  // CLEANUP
  // ==========================================================

  useEffect(() => {
    return () => {
      service.current?.stop();

      service.current =
        null;
    };
  }, []);


  // ==========================================================
  // START / STOP VOICE
  // ==========================================================

  const toggleVoice =
    async () => {
      setError(null);


      // ------------------------------------------------------
      // STOP CURRENT VOICE SESSION
      // ------------------------------------------------------

      if (
        service.current?.isConnected()
      ) {
        service.current.stop();

        service.current =
          null;

        setState(
          "idle"
        );

        return;
      }


      // ------------------------------------------------------
      // START VOICE SESSION
      // ------------------------------------------------------

      try {
        const live =
          new GeminiLiveService({

            // ----------------------------------------------
            // State
            // ----------------------------------------------

            onStateChange:
              (
                newState
              ) => {
                setState(
                  newState
                );
              },


            // ----------------------------------------------
            // User transcription
            // ----------------------------------------------

            onUserMessage:
              (
                text
              ) => {
                console.log(
                  "Voice user message:",
                  text
                );

                addVoiceUserMessage(
                  text
                );
              },


            // ----------------------------------------------
            // Real-time user transcription streaming
            // ----------------------------------------------

            onUserTranscript:
              (
                text
              ) => {
                addVoiceUserMessage(
                  text
                );
              },


            // ----------------------------------------------
            // Assistant transcription
            // ----------------------------------------------

            onAssistantMessage:
              (
                text
              ) => {
                console.log(
                  "Voice assistant response:",
                  text
                );

                addVoiceAssistantMessage(
                  text
                );
              },


            // ----------------------------------------------
            // Real-time streaming transcription delta
            // ----------------------------------------------

            onTranscriptDelta:
              (
                deltaText,
                isDone
              ) => {
                updateStreamingAssistantMessage(
                  deltaText,
                  isDone
                );
              },


            // ----------------------------------------------
            // New conversation
            // ----------------------------------------------

            onConversationCreated:
              (
                id
              ) => {
                console.log(
                  "Voice conversation created:",
                  id
                );

                setVoiceConversationId(
                  id
                );
              },


            // ----------------------------------------------
            // Error
            // ----------------------------------------------

            onError:
              (
                voiceError
              ) => {
                console.error(
                  "Voice error:",
                  voiceError
                );

                setError(
                  voiceError.message
                );

                setState(
                  "error"
                );
              },

          });


        // ----------------------------------------------------
        // VERY IMPORTANT
        //
        // Continue the currently active chat session.
        // If conversationId is null, backend will create
        // a new conversation after the first voice turn.
        // ----------------------------------------------------

        live.setConversationId(
          conversationId
        );


        service.current =
          live;


        await live.start();

      } catch (error) {
        console.error(
          "Unable to start voice:",
          error
        );


        service.current =
          null;


        setState(
          "error"
        );


        setError(
          error instanceof Error
            ? error.message
            : "Unable to start voice mode."
        );
      }
    };


  // ==========================================================
  // ACTIVE STATE
  // ==========================================================

  const isActive =
    state !== "idle";


  // ==========================================================
  // BUTTON LABEL
  // ==========================================================

  const getLabel =
    () => {
      switch (state) {

        case "connecting":
          return "Connecting to WAC AI...";

        case "listening":
          return "Listening...";

        case "speaking":
          return "WAC AI is speaking...";

        case "error":
          return "Voice error";

        default:
          return "Start voice";
      }
    };


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="relative">

      <button
        type="button"
        onClick={toggleVoice}
        aria-label={getLabel()}
        title={getLabel()}
        className={`
          relative
          flex
          h-11
          w-11
          items-center
          justify-center
          rounded-full
          transition-all
          duration-200
          ${
            isActive
              ? "bg-red-500 text-white shadow-lg shadow-red-500/30"
              : "bg-white text-gray-600 hover:bg-gray-100"
          }
        `}
      >

        {state === "connecting" ? (

          <span
            className="
              h-5
              w-5
              animate-spin
              rounded-full
              border-2
              border-gray-300
              border-t-blue-500
            "
          />

        ) : state === "speaking" ? (

          <span
            className="
              animate-pulse
              text-lg
            "
          >
            🔊
          </span>

        ) : (

          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >

            <rect
              x="9"
              y="2"
              width="6"
              height="12"
              rx="3"
            />

            <path
              d="M5 10a7 7 0 0 0 14 0"
            />

            <line
              x1="12"
              y1="19"
              x2="12"
              y2="22"
            />

            <line
              x1="8"
              y1="22"
              x2="16"
              y2="22"
            />

          </svg>

        )}


        {state === "listening" && (

          <span
            className="
              absolute
              inset-0
              animate-ping
              rounded-full
              border-2
              border-red-400
              opacity-50
            "
          />

        )}

      </button>


      {error && (

        <div
          className="
            absolute
            right-0
            top-14
            z-50
            w-64
            rounded-xl
            border
            border-red-200
            bg-white
            p-3
            text-xs
            text-red-600
            shadow-lg
          "
        >
          {error}
        </div>

      )}

    </div>
  );
}
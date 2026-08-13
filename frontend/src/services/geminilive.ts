import {
  GoogleGenAI,
  Modality,
} from "@google/genai";


const API_BASE_URL =
  "http://127.0.0.1:8000";


export type VoiceState =
  | "idle"
  | "connecting"
  | "listening"
  | "speaking"
  | "error";


interface GeminiLiveCallbacks {

  onStateChange?: (
    state: VoiceState
  ) => void;

  onUserMessage?: (
    text: string
  ) => void;

  onAssistantMessage?: (
    text: string
  ) => void;

  onConversationCreated?: (
    conversationId: string
  ) => void;

  onError?: (
    error: Error
  ) => void;
}


export class GeminiLiveService {

  // ==========================================================
  // GEMINI SESSION
  // ==========================================================

  private session: any = null;


  // ==========================================================
  // AUDIO
  // ==========================================================

  private inputAudioContext:
    AudioContext | null = null;

  private outputAudioContext:
    AudioContext | null = null;

  private microphoneStream:
    MediaStream | null = null;

  private microphoneSource:
    MediaStreamAudioSourceNode | null = null;

  private processor:
    ScriptProcessorNode | null = null;

  private silentGain:
    GainNode | null = null;


  // ==========================================================
  // AUDIO SCHEDULING
  // ==========================================================

  private nextAudioTime = 0;


  // ==========================================================
  // CALLBACKS
  // ==========================================================

  private callbacks:
    GeminiLiveCallbacks;


  // ==========================================================
  // CONVERSATION
  // ==========================================================

  private conversationId:
    string | null = null;


  // ==========================================================
  // TRANSCRIPTS
  // ==========================================================

  private userTranscript = "";

  private assistantTranscript = "";


  // ==========================================================
  // CONSTRUCTOR
  // ==========================================================

  constructor(
    callbacks: GeminiLiveCallbacks = {}
  ) {
    this.callbacks =
      callbacks;
  }


  // ==========================================================
  // SET CONVERSATION
  // ==========================================================

  setConversationId(
    conversationId: string | null
  ) {
    this.conversationId =
      conversationId;
  }


  // ==========================================================
  // STATE
  // ==========================================================

  private setState(
    state: VoiceState
  ) {
    console.log(
      "Gemini Live state:",
      state
    );

    this.callbacks
      .onStateChange?.(
        state
      );
  }


  // ==========================================================
  // GET EPHEMERAL TOKEN
  // ==========================================================

  private async getToken() {

    const response =
      await fetch(
        `${API_BASE_URL}/voice/token`
      );


    if (!response.ok) {
      throw new Error(
        "Unable to start WAC AI voice service."
      );
    }


    return response.json();
  }


  // ==========================================================
  // SAVE VOICE CONVERSATION
  // ==========================================================

  private async saveVoiceConversation(
    userMessage: string,
    assistantMessage: string
  ) {

    const response =
      await fetch(
        `${API_BASE_URL}/voice/message`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({

            conversation_id:
              this.conversationId,

            user_message:
              userMessage,

            assistant_message:
              assistantMessage,

          }),
        }
      );


    if (!response.ok) {

      let detail =
        "Failed to save voice conversation.";

      try {
        const errorData =
          await response.json();

        if (
          typeof errorData?.detail ===
          "string"
        ) {
          detail =
            errorData.detail;
        }
      } catch {
        // Ignore invalid error body.
      }

      throw new Error(
        detail
      );
    }


    const data =
      await response.json();


    if (
      data?.conversation_id
    ) {

      const newConversationId =
        String(
          data.conversation_id
        );


      this.conversationId =
        newConversationId;


      this.callbacks
        .onConversationCreated?.(
          newConversationId
        );
    }


    return data;
  }


  // ==========================================================
  // START
  // ==========================================================

  async start(): Promise<void> {

    if (this.session) {
      return;
    }


    try {

      this.setState(
        "connecting"
      );


      // ------------------------------------------------------
      // MICROPHONE
      // ------------------------------------------------------

      this.microphoneStream =
        await navigator
          .mediaDevices
          .getUserMedia({

            audio: {
              channelCount: 1,

              echoCancellation:
                true,

              noiseSuppression:
                true,

              autoGainControl:
                true,
            },

            video: false,
          });


      // ------------------------------------------------------
      // EPHEMERAL TOKEN
      // ------------------------------------------------------

      const tokenData =
        await this.getToken();


      if (
        !tokenData?.token ||
        !tokenData?.model
      ) {
        throw new Error(
          "Invalid voice token response."
        );
      }


      const ai =
        new GoogleGenAI({
          apiKey:
            tokenData.token,
        });


      // ------------------------------------------------------
      // INPUT AUDIO
      // ------------------------------------------------------

      this.inputAudioContext =
        new AudioContext({
          sampleRate: 16000,
        });


      await this.inputAudioContext
        .resume();


      // ------------------------------------------------------
      // OUTPUT AUDIO
      // ------------------------------------------------------

      this.outputAudioContext =
        new AudioContext({
          sampleRate: 24000,
        });


      await this.outputAudioContext
        .resume();


      this.nextAudioTime =
        this.outputAudioContext
          .currentTime;


      // ------------------------------------------------------
      // CONNECT GEMINI LIVE
      // ------------------------------------------------------

      this.session =
        await ai.live.connect({

          model:
            tokenData.model,

          config: {

            responseModalities: [
              Modality.AUDIO,
            ],

            systemInstruction: {
              parts: [
                {
                  text: `
You are WAC AI, the official voice assistant for Web and Craft.

Your ONLY purpose is to answer questions related to Web and Craft.

You may discuss:

- WAC company information
- WAC services
- WAC technologies
- WAC AI and software capabilities
- WAC industries
- WAC clients and case studies
- WAC careers
- WAC leadership
- WAC contact and office information

IMPORTANT:

If a question is unrelated to Web and Craft, do not answer the unrelated question.

Instead respond with:

"I'm WAC AI, a specialized AI assistant for Web and Craft. I can only help with questions related to Web and Craft, its services, technologies, projects, careers, and company information."

Do not answer general questions about unrelated topics.

Do not invent information about WAC.

VOICE RESPONSE RULES:

- Speak naturally.
- Be concise.
- Do not use Markdown.
- Do not use hashtags.
- Do not use bullet points.
- Do not use headings.
- Do not repeat the user's question.
                  `,
                },
              ],
            },

            inputAudioTranscription: {},

            outputAudioTranscription: {},

            contextWindowCompression: {
              slidingWindow: {},
            },
          },


          callbacks: {

            // ------------------------------------------------
            // OPEN
            // ------------------------------------------------

            onopen: () => {

              console.log(
                "Gemini Live connected."
              );

              this.setState(
                "listening"
              );
            },


            // ------------------------------------------------
            // MESSAGE
            // ------------------------------------------------

            onmessage: (
              message: any
            ) => {

              this.handleMessage(
                message
              );
            },


            // ------------------------------------------------
            // ERROR
            // ------------------------------------------------

            onerror: (
              event: any
            ) => {

              console.error(
                "Gemini Live error:",
                event
              );


              const error =
                new Error(
                  event?.message ||
                  "Gemini Live connection error."
                );


              this.callbacks
                .onError?.(
                  error
                );


              this.setState(
                "error"
              );
            },


            // ------------------------------------------------
            // CLOSE
            // ------------------------------------------------

            onclose: (
              event: CloseEvent
            ) => {

              console.log(
                "Gemini Live closed:",
                event.reason
              );


              this.cleanup();


              this.setState(
                "idle"
              );
            },
          },
        });


      // ------------------------------------------------------
      // START MICROPHONE
      // ------------------------------------------------------

      this.startMicrophone();

    } catch (error) {

      console.error(
        "Failed to start Gemini Live:",
        error
      );


      this.cleanup();


      const normalizedError =
        error instanceof Error
          ? error
          : new Error(
              "Unable to start voice mode."
            );


      this.callbacks
        .onError?.(
          normalizedError
        );


      this.setState(
        "error"
      );


      throw normalizedError;
    }
  }


  // ==========================================================
  // MICROPHONE
  // ==========================================================

  private startMicrophone() {

    if (
      !this.inputAudioContext ||
      !this.microphoneStream ||
      !this.session
    ) {
      return;
    }


    this.microphoneSource =
      this.inputAudioContext
        .createMediaStreamSource(
          this.microphoneStream
        );


    this.processor =
      this.inputAudioContext
        .createScriptProcessor(
          4096,
          1,
          1
        );


    this.silentGain =
      this.inputAudioContext
        .createGain();


    this.silentGain.gain.value =
      0;


    this.microphoneSource
      .connect(
        this.processor
      );


    this.processor
      .connect(
        this.silentGain
      );


    this.silentGain
      .connect(
        this.inputAudioContext
          .destination
      );


    this.processor.onaudioprocess =
      (
        event: AudioProcessingEvent
      ) => {

        if (!this.session) {
          return;
        }


        const input =
          event.inputBuffer
            .getChannelData(0);


        const pcm =
          this.floatTo16BitPCM(
            input
          );


        const base64 =
          this.arrayBufferToBase64(
            pcm.buffer
          );


        this.session
          .sendRealtimeInput({

            audio: {
              data: base64,

              mimeType:
                "audio/pcm;rate=16000",
            },

          });
      };
  }


  // ==========================================================
  // FLOAT32 → INT16 PCM
  // ==========================================================

  private floatTo16BitPCM(
    input: Float32Array
  ): Int16Array {

    const output =
      new Int16Array(
        input.length
      );


    for (
      let i = 0;
      i < input.length;
      i++
    ) {

      const sample =
        Math.max(
          -1,
          Math.min(
            1,
            input[i]
          )
        );


      output[i] =
        sample < 0
          ? sample * 0x8000
          : sample * 0x7fff;
    }


    return output;
  }


  // ==========================================================
  // ARRAY BUFFER → BASE64
  // ==========================================================

  private arrayBufferToBase64(
    buffer: ArrayBufferLike
  ): string {

    const bytes =
      new Uint8Array(
        buffer
      );


    let binary = "";


    const chunkSize =
      0x8000;


    for (
      let i = 0;
      i < bytes.length;
      i += chunkSize
    ) {

      const chunk =
        bytes.subarray(
          i,
          Math.min(
            i + chunkSize,
            bytes.length
          )
        );


      binary +=
        String.fromCharCode(
          ...chunk
        );
    }


    return window.btoa(
      binary
    );
  }


  // ==========================================================
  // HANDLE GEMINI MESSAGE
  // ==========================================================

  private handleMessage(
    message: any
  ) {

    const serverContent =
      message?.serverContent;


    if (!serverContent) {
      return;
    }


    // ========================================================
    // USER TRANSCRIPTION
    // ========================================================

    if (
      serverContent.inputTranscription
    ) {

      const text =
        serverContent
          .inputTranscription
          .text;


      if (text) {

        this.userTranscript +=
          text;


        console.log(
          "USER:",
          this.userTranscript
        );
      }
    }


    // ========================================================
    // ASSISTANT TRANSCRIPTION
    // ========================================================

    if (
      serverContent.outputTranscription
    ) {

      const text =
        serverContent
          .outputTranscription
          .text;


      if (text) {

        this.assistantTranscript +=
          text;


        console.log(
          "WAC AI:",
          this.assistantTranscript
        );
      }
    }


    // ========================================================
    // INTERRUPTION
    // ========================================================

    if (
      serverContent.interrupted
    ) {

      console.log(
        "User interrupted WAC AI."
      );


      if (
        this.outputAudioContext
      ) {

        this.nextAudioTime =
          this.outputAudioContext
            .currentTime;
      }


      this.setState(
        "listening"
      );


      return;
    }


    // ========================================================
    // AUDIO RESPONSE
    // ========================================================

    const parts =
      serverContent
        ?.modelTurn
        ?.parts;


    if (parts) {

      for (
        const part of parts
      ) {

        const audioData =
          part
            ?.inlineData
            ?.data;


        if (audioData) {

          this.setState(
            "speaking"
          );


          this.playAudio(
            audioData
          );
        }
      }
    }


    // ========================================================
    // TURN COMPLETE
    // ========================================================

    if (
      serverContent.turnComplete
    ) {

      console.log(
        "Gemini turn completed."
      );


      const userMessage =
        this.userTranscript.trim();


      const assistantMessage =
        this.assistantTranscript.trim();


      // ------------------------------------------------------
      // DISPLAY USER MESSAGE
      // ------------------------------------------------------

      if (userMessage) {

        this.callbacks
          .onUserMessage?.(
            userMessage
          );
      }


      // ------------------------------------------------------
      // DISPLAY ASSISTANT MESSAGE
      // ------------------------------------------------------

      if (assistantMessage) {

        this.callbacks
          .onAssistantMessage?.(
            assistantMessage
          );
      }


      // ------------------------------------------------------
      // SAVE BOTH MESSAGES
      // ------------------------------------------------------

      if (
        userMessage &&
        assistantMessage
      ) {

        this.saveVoiceConversation(
          userMessage,
          assistantMessage
        )
          .catch(
            error => {

              console.error(
                "Failed to persist voice conversation:",
                error
              );

            }
          );
      }


      // ------------------------------------------------------
      // RESET TRANSCRIPTS
      // ------------------------------------------------------

      this.userTranscript =
        "";

      this.assistantTranscript =
        "";


      // ------------------------------------------------------
      // RETURN TO LISTENING
      // ------------------------------------------------------

      if (
        !this.outputAudioContext ||
        this.nextAudioTime <=
          this.outputAudioContext
            .currentTime
      ) {

        this.setState(
          "listening"
        );
      }
    }
  }


  // ==========================================================
  // PLAY GEMINI AUDIO
  // ==========================================================

  private async playAudio(
    base64Audio: string
  ) {

    if (
      !this.outputAudioContext
    ) {
      return;
    }


    if (
      this.outputAudioContext
        .state === "suspended"
    ) {

      await this.outputAudioContext
        .resume();
    }


    const binary =
      window.atob(
        base64Audio
      );


    const bytes =
      new Uint8Array(
        binary.length
      );


    for (
      let i = 0;
      i < binary.length;
      i++
    ) {

      bytes[i] =
        binary.charCodeAt(i);
    }


    const pcm =
      new Int16Array(
        bytes.buffer
      );


    const float32 =
      new Float32Array(
        pcm.length
      );


    for (
      let i = 0;
      i < pcm.length;
      i++
    ) {

      float32[i] =
        pcm[i] / 32768;
    }


    // --------------------------------------------------------
    // Gemini native audio = 24 kHz
    // --------------------------------------------------------

    const audioBuffer =
      this.outputAudioContext
        .createBuffer(
          1,
          float32.length,
          24000
        );


    audioBuffer
      .getChannelData(0)
      .set(float32);


    const source =
      this.outputAudioContext
        .createBufferSource();


    source.buffer =
      audioBuffer;


    source.connect(
      this.outputAudioContext
        .destination
    );


    const startTime =
      Math.max(
        this.outputAudioContext
          .currentTime,

        this.nextAudioTime
      );


    source.start(
      startTime
    );


    this.nextAudioTime =
      startTime +
      audioBuffer.duration;


    source.onended =
      () => {

        if (
          this.outputAudioContext &&
          this.nextAudioTime <=
            this.outputAudioContext
              .currentTime
        ) {

          this.setState(
            "listening"
          );
        }
      };
  }


  // ==========================================================
  // STOP
  // ==========================================================

  stop() {

    console.log(
      "Stopping Gemini Live."
    );


    if (this.session) {

      try {

        this.session.close();

      } catch (error) {

        console.warn(
          "Error closing Live session:",
          error
        );
      }
    }


    this.cleanup();


    this.setState(
      "idle"
    );
  }


  // ==========================================================
  // CLEANUP
  // ==========================================================

  private cleanup() {

    // --------------------------------------------------------
    // MICROPHONE
    // --------------------------------------------------------

    if (
      this.microphoneStream
    ) {

      this.microphoneStream
        .getTracks()
        .forEach(
          track => {
            track.stop();
          }
        );


      this.microphoneStream =
        null;
    }


    // --------------------------------------------------------
    // AUDIO NODES
    // --------------------------------------------------------

    try {

      this.processor
        ?.disconnect();

      this.microphoneSource
        ?.disconnect();

      this.silentGain
        ?.disconnect();

    } catch {
      // Ignore cleanup errors.
    }


    this.processor =
      null;

    this.microphoneSource =
      null;

    this.silentGain =
      null;


    // --------------------------------------------------------
    // INPUT AUDIO CONTEXT
    // --------------------------------------------------------

    if (
      this.inputAudioContext
    ) {

      this.inputAudioContext
        .close()
        .catch(() => {});


      this.inputAudioContext =
        null;
    }


    // --------------------------------------------------------
    // OUTPUT AUDIO CONTEXT
    // --------------------------------------------------------

    if (
      this.outputAudioContext
    ) {

      this.outputAudioContext
        .close()
        .catch(() => {});


      this.outputAudioContext =
        null;
    }


    // --------------------------------------------------------
    // SESSION
    // --------------------------------------------------------

    this.session =
      null;


    this.nextAudioTime =
      0;


    // --------------------------------------------------------
    // TRANSCRIPTS
    // --------------------------------------------------------

    this.userTranscript =
      "";

    this.assistantTranscript =
      "";
  }


  // ==========================================================
  // CONNECTION STATUS
  // ==========================================================

  isConnected(): boolean {

    return (
      this.session !== null
    );
  }
}
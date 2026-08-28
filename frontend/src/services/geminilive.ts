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
  // ACTIVE AUDIO SOURCES
  //
  // Every Gemini audio chunk creates an
  // AudioBufferSourceNode.
  //
  // We track every source so that when the user interrupts
  // Gemini, all already-scheduled audio can be stopped.
  // ==========================================================

  private activeAudioSources =
    new Set<AudioBufferSourceNode>();


  // ==========================================================
  // AUDIO GENERATION
  //
  // Every interruption increments this value.
  //
  // Audio belonging to an older generation will be ignored.
  // ==========================================================

  private audioGeneration = 0;


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
  // SET CONVERSATION ID
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
      // RESET AUDIO GENERATION
      // ------------------------------------------------------

      this.audioGeneration++;


      // ======================================================
      // CONNECT GEMINI LIVE
      // ======================================================

      this.session =
        await ai.live.connect({

          model:
            tokenData.model,

          config: {

            // ------------------------------------------------
            // AUDIO
            // ------------------------------------------------

            responseModalities: [
              Modality.AUDIO,
            ],


            // ------------------------------------------------
            // TRANSCRIPTION
            // ------------------------------------------------

            inputAudioTranscription: {},

            outputAudioTranscription: {},


            // ------------------------------------------------
            // WAC FUNCTION CALLING
            //
            // Backend /voice/token returns:
            //
            // {
            //   tools: [...]
            // }
            //
            // These tools are supplied to Gemini Live.
            // ------------------------------------------------

            tools:
              tokenData.tools || [],


            // ------------------------------------------------
            // WAC SYSTEM INSTRUCTION
            // ------------------------------------------------

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

For factual information about WAC, use the
search_wac_knowledge tool.

The tool searches WAC's official knowledge base.

Never invent information about WAC.

Never use general world knowledge to answer
questions about WAC.

If the question is unrelated to Web and Craft,
do not answer it.

Instead say:

"I'm WAC AI, a specialized AI assistant for Web and Craft. I can only help with questions related to Web and Craft, its services, technologies, projects, careers, and company information."

VOICE RESPONSE RULES:

- Speak naturally.
- Be concise.
- Do not use Markdown.
- Do not use hashtags.
- Do not use bullet points.
- Do not use headings.
- Do not repeat the user's question.

TOOL RULE:

When a user asks about WAC information that
requires factual knowledge, call
search_wac_knowledge before answering.

Use the returned information as the source
of truth.

If the tool does not find reliable information,
say that you could not find the information
in WAC's current knowledge base.
                  `,
                },
              ],
            },


            // ------------------------------------------------
            // CONTEXT WINDOW
            // ------------------------------------------------

            contextWindowCompression: {
              slidingWindow: {},
            },
          },


          // ==================================================
          // CALLBACKS
          // ==================================================

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

              // IMPORTANT:
              //
              // handleMessage is async because Gemini Live
              // function calls require asynchronous backend
              // RAG execution.
              //
              this.handleMessage(
                message
              ).catch(
                error => {

                  console.error(
                    "Gemini Live message handling failed:",
                    error
                  );

                }
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
  // GEMINI LIVE TOOL EXECUTION
  //
  // Gemini Live requests:
  //
  // search_wac_knowledge(...)
  //
  // This method sends that request to our backend.
  //
  // Browser NEVER directly accesses MongoDB or RAG internals.
  // ==========================================================

  private async executeToolCall(
    name: string,
    arguments_: Record<string, unknown>
  ): Promise<Record<string, unknown>> {

    console.log(
      "Gemini Live tool call:",
      name,
      arguments_
    );


    // --------------------------------------------------------
    // SECURITY:
    //
    // Only explicitly approved tools are allowed.
    // --------------------------------------------------------

    if (
      name !==
      "search_wac_knowledge"
    ) {

      return {
        success: false,

        error:
          `Unsupported WAC tool: ${name}`,
      };
    }


    // --------------------------------------------------------
    // Validate query
    // --------------------------------------------------------

    const query =
      arguments_?.query;


    if (
      typeof query !== "string" ||
      !query.trim()
    ) {

      return {

        success: false,

        error:
          "The search query is required.",
      };
    }


    try {

      // ------------------------------------------------------
      // Execute backend RAG tool
      // ------------------------------------------------------

      const response =
        await fetch(
          `${API_BASE_URL}/voice/tool`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({

              name,

              arguments: {
                query:
                  query.trim(),
              },

              conversation_id:
                this.conversationId,

            }),
          }
        );


      // ------------------------------------------------------
      // HTTP ERROR
      // ------------------------------------------------------

      if (!response.ok) {

        let detail =
          "WAC knowledge search failed.";

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
          // Ignore malformed error body.
        }


        return {

          success: false,

          error:
            detail,
        };
      }


      // ------------------------------------------------------
      // Successful RAG response
      // ------------------------------------------------------

      const result =
        await response.json();


      console.log(
        "WAC RAG tool result:",
        result
      );


      return result;

    } catch (error) {

      console.error(
        "Voice tool execution failed:",
        error
      );


      return {

        success: false,

        error:
          "WAC knowledge search failed.",
      };
    }
  }


  // ==========================================================
  // HANDLE GEMINI LIVE TOOL CALL
  //
  // Flow:
  //
  // Gemini
  //   ↓
  // toolCall
  //   ↓
  // executeToolCall()
  //   ↓
  // /voice/tool
  //   ↓
  // RAGService
  //   ↓
  // result
  //   ↓
  // sendToolResponse()
  //   ↓
  // Gemini
  // ==========================================================

  private async handleToolCall(
    toolCall: any
  ): Promise<void> {

    console.log(
      "Gemini Live toolCall event:",
      toolCall
    );


    const functionCalls =
      toolCall?.functionCalls;


    if (
      !Array.isArray(functionCalls) ||
      functionCalls.length === 0
    ) {

      console.warn(
        "Gemini Live toolCall contained no function calls."
      );

      return;
    }


    const functionResponses: Array<{
      id?: string;
      name: string;
      response: Record<string, unknown>;
    }> = [];


    // --------------------------------------------------------
    // Execute every requested function
    // --------------------------------------------------------

    for (
      const functionCall of functionCalls
    ) {

      const name =
        functionCall?.name;


      const args =
        functionCall?.args || {};


      const callId =
        functionCall?.id;


      if (!name) {

        console.warn(
          "Gemini Live returned tool call without name."
        );

        continue;
      }


      console.log(
        "Executing Gemini Live function:",
        {
          id: callId,
          name,
          args,
        }
      );


      try {

        const result =
          await this.executeToolCall(
            name,
            args
          );


        functionResponses.push({

          id:
            callId,

          name,

          response:
            result,
        });

      } catch (error) {

        console.error(
          "Tool execution failed:",
          error
        );


        functionResponses.push({

          id:
            callId,

          name,

          response: {

            success:
              false,

            error:
              "The WAC knowledge search failed.",
          },
        });
      }
    }


    // --------------------------------------------------------
    // Verify Live session still exists
    // --------------------------------------------------------

    if (
      !this.session
    ) {

      console.warn(
        "Gemini Live session no longer exists. Tool response skipped."
      );

      return;
    }


    if (
      functionResponses.length === 0
    ) {

      return;
    }


    // --------------------------------------------------------
    // Send function response back to Gemini
    //
    // Gemini will continue the conversation and generate
    // the final spoken answer.
    // --------------------------------------------------------

    try {

      this.session.sendToolResponse({

        functionResponses:
          functionResponses,

      });


      console.log(
        "WAC RAG tool response sent to Gemini Live."
      );

    } catch (error) {

      console.error(
        "Failed to send tool response to Gemini Live:",
        error
      );
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

        if (
          !this.session
        ) {

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


        try {

          this.session
            .sendRealtimeInput({

              audio: {

                data:
                  base64,

                mimeType:
                  "audio/pcm;rate=16000",
              },

            });

        } catch (error) {

          console.warn(
            "Failed to send realtime audio:",
            error
          );
        }
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

  private async handleMessage(
    message: any
  ): Promise<void> {

    // ========================================================
    // FUNCTION CALL
    //
    // IMPORTANT:
    //
    // This MUST be handled before serverContent because
    // function calls are separate Live events.
    // ========================================================

    if (
      message?.toolCall
    ) {

      console.log(
        "Gemini Live requested WAC tool."
      );


      await this.handleToolCall(
        message.toolCall
      );


      return;
    }


    // ========================================================
    // SERVER CONTENT
    // ========================================================

    const serverContent =
      message?.serverContent;


    if (
      !serverContent
    ) {

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


      // ------------------------------------------------------
      // Stop ALL scheduled audio.
      // ------------------------------------------------------

      this.stopCurrentAudio();


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


    if (
      parts
    ) {

      for (
        const part of parts
      ) {

        const audioData =
          part
            ?.inlineData
            ?.data;


        if (
          audioData
        ) {

          this.setState(
            "speaking"
          );


          await this.playAudio(
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

      if (
        userMessage
      ) {

        this.callbacks
          .onUserMessage?.(
            userMessage
          );
      }


      // ------------------------------------------------------
      // DISPLAY ASSISTANT MESSAGE
      // ------------------------------------------------------

      if (
        assistantMessage
      ) {

        this.callbacks
          .onAssistantMessage?.(
            assistantMessage
          );
      }


      // ------------------------------------------------------
      // SAVE CONVERSATION
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
  // STOP CURRENT AUDIO
  //
  // MAIN INTERRUPTION FIX
  // ==========================================================

  private stopCurrentAudio(): void {

    console.log(
      "Stopping all active Gemini audio sources."
    );


    // --------------------------------------------------------
    // Invalidate old audio generation.
    // --------------------------------------------------------

    this.audioGeneration++;


    // --------------------------------------------------------
    // Stop every active source.
    // --------------------------------------------------------

    for (
      const source of this.activeAudioSources
    ) {

      try {

        source.stop();

      } catch {
        // Source may already have stopped.
      }
    }


    // --------------------------------------------------------
    // Clear source tracking.
    // --------------------------------------------------------

    this.activeAudioSources.clear();


    // --------------------------------------------------------
    // Reset scheduler.
    // --------------------------------------------------------

    if (
      this.outputAudioContext
    ) {

      this.nextAudioTime =
        this.outputAudioContext
          .currentTime;

    } else {

      this.nextAudioTime =
        0;
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


    // --------------------------------------------------------
    // Capture current generation.
    // --------------------------------------------------------

    const generation =
      this.audioGeneration;


    // --------------------------------------------------------
    // Decode Base64
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // Gemini Live returns signed 16-bit PCM.
    // --------------------------------------------------------

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
    // Gemini native audio = 24 kHz.
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


    // --------------------------------------------------------
    // Check generation before scheduling.
    // --------------------------------------------------------

    if (
      generation !==
      this.audioGeneration
    ) {

      console.log(
        "Ignoring stale Gemini audio chunk."
      );


      return;
    }


    const source =
      this.outputAudioContext
        .createBufferSource();


    source.buffer =
      audioBuffer;


    source.connect(
      this.outputAudioContext
        .destination
    );


    // --------------------------------------------------------
    // Track source.
    // --------------------------------------------------------

    this.activeAudioSources.add(
      source
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


    // --------------------------------------------------------
    // Remove source after playback.
    // --------------------------------------------------------

    source.onended =
      () => {

        this.activeAudioSources.delete(
          source
        );


        if (
          this.activeAudioSources.size ===
          0
        ) {

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


    // --------------------------------------------------------
    // Stop audio FIRST.
    // --------------------------------------------------------

    this.stopCurrentAudio();


    // --------------------------------------------------------
    // Close Live session.
    // --------------------------------------------------------

    if (
      this.session
    ) {

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
    // Stop audio.
    // --------------------------------------------------------

    this.stopCurrentAudio();


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


    // --------------------------------------------------------
    // AUDIO
    // --------------------------------------------------------

    this.nextAudioTime =
      0;


    this.activeAudioSources.clear();


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
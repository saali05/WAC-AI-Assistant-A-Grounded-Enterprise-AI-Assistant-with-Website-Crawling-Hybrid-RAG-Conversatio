const API_BASE_URL =
  "http://127.0.0.1:8000";


// ============================================================
// TYPES
// ============================================================

export interface Conversation {
  id: string;
  title: string;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ConversationDetail {
  id?: string;
  title?: string;
  messages: ConversationMessage[];
}


// ============================================================
// GET ALL CONVERSATIONS
// ============================================================

export async function getConversations(): Promise<
  Conversation[]
> {
  const response =
    await fetch(
      `${API_BASE_URL}/sessions`
    );


  if (!response.ok) {
    throw new Error(
      "Failed to load conversations."
    );
  }


  const data =
    await response.json();


  /*
   * Supports both:
   *
   * [
   *   { id, title }
   * ]
   *
   * and:
   *
   * {
   *   sessions: [...]
   * }
   *
   * and:
   *
   * {
   *   conversations: [...]
   * }
   */

  const items =
    Array.isArray(data)
      ? data
      : Array.isArray(data.sessions)
        ? data.sessions
        : Array.isArray(
            data.conversations
          )
          ? data.conversations
          : [];


  return items.map(
    (item: any) => ({
      id:
        String(
          item.id ??
          item._id ??
          item.session_id
        ),

      title:
        item.title ??
        item.name ??
        "New conversation",
    })
  );
}


// ============================================================
// GET SINGLE CONVERSATION
// ============================================================

export async function getConversation(
  id: string
): Promise<ConversationDetail> {
  const response =
    await fetch(
      `${API_BASE_URL}/history/${encodeURIComponent(
        id
      )}`
    );


  if (!response.ok) {
    throw new Error(
      "Failed to load conversation."
    );
  }


  const data =
    await response.json();


  /*
   * Expected:
   *
   * {
   *   messages: [...]
   * }
   */

  return {
    id:
      data.id ??
      data.session_id ??
      id,

    title:
      data.title ??
      "Conversation",

    messages:
      Array.isArray(
        data.messages
      )
        ? data.messages.map(
            (message: any) => ({
              role:
                message.role ===
                "assistant"
                  ? "assistant"
                  : "user",

              content:
                message.content ??
                message.message ??
                "",
            })
          )
        : [],
  };
}
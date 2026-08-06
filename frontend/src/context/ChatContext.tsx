import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

import type { Provider } from "../types/chat";

import {
  sendMessage,
} from "../api/chat";

import {
  getConversation,
  getConversations,
} from "../api/conversation";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface Conversation {
  id: string;
  title: string;
}

interface ChatContextType {

  messages: Message[];

  conversations: Conversation[];

  loading: boolean;

  conversationId: string | null;

  send: (
    
    message: string,
    provider: Provider
  ) => Promise<void>;

  loadConversation: (
    id: string
  ) => Promise<void>;

  reloadConversations: () => Promise<void>;

  newChat: () => void;

}

const ChatContext =
createContext<ChatContextType | null>(
null
);

export function ChatProvider({
children,
}:{
children:ReactNode;
}){

const [messages,setMessages]=
useState<Message[]>([]);

const [conversations,setConversations]=
useState<Conversation[]>([]);

const [loading,setLoading]=
useState(false);

const [conversationId,setConversationId]=
useState<string|null>(null);

const reloadConversations=
async()=>{

const data=
await getConversations();

setConversations(data);

};

const newChat=()=>{

setConversationId(null);

setMessages([]);

};

const loadConversation=
async(id:string)=>{

const data=
await getConversation(id);

setConversationId(id);

setMessages(data.messages);

};

const send=
async(
text:string,
provider:Provider,
)=>{

if(!text.trim()) return;

setMessages(prev=>[
...prev,
{
role:"user",
content:text,
},
]);

setLoading(true);

try{

const response=
await sendMessage({

conversation_id:
conversationId ?? undefined,

provider,

message:text,

});

if(!conversationId){

setConversationId(
response.conversation_id
);

await reloadConversations();

}
console.log("Backend response:", response);

setMessages(prev => {
  const updated = [
    ...prev,
    {
      role: "assistant" as const,
      content: response.response,
    },
  ];

  console.log("Updated messages:", updated);

  return updated;
});

}catch (error) {

    const message =
        error instanceof Error
            ? error.message
            : "Something went wrong.";

    setMessages(prev => [
        ...prev,
        {
            role: "assistant",
            content: message,
        },
    ]);

}finally{

setLoading(false);

}

};

return(

<ChatContext.Provider

value={{

messages,

conversations,

loading,

conversationId,

send,

loadConversation,

reloadConversations,

newChat,

}}

>

{children}

</ChatContext.Provider>

);

}

export function useChat(){

const context=
useContext(ChatContext);

if(!context){

throw new Error(
"useChat must be used inside ChatProvider"
);

}

return context;

}
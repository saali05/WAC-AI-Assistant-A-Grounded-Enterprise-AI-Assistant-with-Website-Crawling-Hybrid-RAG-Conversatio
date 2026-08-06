import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";



interface ChatWindowProps {
  messages: {
    role: "user" | "assistant";
    content: string;
  }[];
}

export default function ChatWindow({
  messages,
}: ChatWindowProps) {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-8 py-10">

      {messages.map((message, index) => (
        <div
          key={index}
          className={
            message.role === "user"
              ? "ml-auto max-w-[75%] rounded-3xl bg-blue-600 px-5 py-3 text-white"
              : "mr-auto max-w-[75%] rounded-3xl bg-white px-5 py-3 shadow"
          }
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>
      ))}

    </div>
  );
}
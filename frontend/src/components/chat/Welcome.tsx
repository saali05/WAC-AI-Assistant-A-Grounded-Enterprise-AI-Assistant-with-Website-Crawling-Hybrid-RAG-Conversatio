import { Briefcase, Cpu, Building2, PhoneCall, Sparkles, ArrowUpRight } from "lucide-react";

interface WelcomeProps {
  onSelectPrompt?: (promptText: string) => void;
}

export default function Welcome({ onSelectPrompt }: WelcomeProps) {
  const promptSuggestions = [
    {
      icon: Briefcase,
      color: "from-red-500 to-rose-600",
      title: "Services & Solutions",
      prompt: "What core services, custom web development, and digital transformation solutions does WebAndCrafts offer?",
    },
    {
      icon: Cpu,
      color: "from-violet-500 to-indigo-600",
      title: "Tech Stack & Innovation",
      prompt: "What modern technology stacks, cloud architectures, and AI frameworks does WebAndCrafts specialize in?",
    },
    {
      icon: Building2,
      color: "from-amber-500 to-orange-600",
      title: "Enterprise Portfolio",
      prompt: "Can you share notable client success stories, enterprise solutions, and projects delivered by WebAndCrafts?",
    },
    {
      icon: PhoneCall,
      color: "from-emerald-500 to-teal-600",
      title: "Consultation & Contact",
      prompt: "How can I request a project estimate or schedule a digital transformation consultation with WebAndCrafts?",
    },
  ];

  return (
    <div className="flex flex-col items-center text-center max-w-3xl mx-auto px-4">
      {/* Top Brand Pill */}
      <div className="inline-flex items-center gap-2 rounded-full border border-red-500/25 bg-gradient-to-r from-red-950/40 via-rose-950/20 to-slate-900/40 px-4 py-1.5 backdrop-blur-md shadow-lg shadow-red-950/20 mb-6">
        <Sparkles size={14} className="text-red-400 animate-pulse" />
        <span className="text-xs font-semibold tracking-wide text-red-200 uppercase">
          Crafting Digital Intelligence
        </span>
      </div>

      {/* Main Headline */}
      <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
        Empowering Ideas Into{" "}
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-rose-400 to-violet-400">
          <br />Digital Crafts
        </span>
      </h1>

      {/* Subtitle */}
      <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-xl leading-relaxed">
        Experience WebAndCrafts Next-Gen AI — Engineered with precision, enterprise power, and elegant execution.
      </p>

      {/* Interactive Prompt Cards Grid */}
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full text-left">
        {promptSuggestions.map((item, index) => {
          const Icon = item.icon;
          return (
            <button
              key={index}
              type="button"
              onClick={() => onSelectPrompt?.(item.prompt)}
              className="
                group
                relative
                flex
                flex-col
                justify-between
                p-4
                rounded-2xl
                wac-glass
                wac-glass-hover
                cursor-pointer
                text-left
              "
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br ${item.color} text-white shadow-md`}>
                    <Icon size={16} />
                  </div>
                  <ArrowUpRight
                    size={16}
                    className="text-slate-500 group-hover:text-red-400 transition-colors group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
                  />
                </div>
                <h3 className="text-sm font-semibold text-white group-hover:text-red-300 transition-colors">
                  {item.title}
                </h3>
                <p className="mt-1 text-xs text-slate-400 line-clamp-2 leading-relaxed">
                  "{item.prompt}"
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
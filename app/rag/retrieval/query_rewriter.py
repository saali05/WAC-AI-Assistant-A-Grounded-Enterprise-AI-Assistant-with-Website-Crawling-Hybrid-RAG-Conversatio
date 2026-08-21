import re


class QueryRewriter:
    """Conversational query rewriter using session memory context."""

    @staticmethod
    def rewrite(user_message: str, conversation_history: str = "") -> str:
        """
        Rewrite user query to make it standalone for retrieval.
        Leaves original user message untouched in conversation memory.
        """
        if not conversation_history or not conversation_history.strip():
            return user_message.strip()

        clean_user = user_message.strip()
        lower_user = clean_user.lower()

        # Pronouns or underspecified follow-up indicators
        follow_up_triggers = ["what about", "tell me more", "how about", "where", "pricing", "details", "them", "it", "this", "that"]

        is_follow_up = any(lower_user.startswith(t) or f" {t} " in f" {lower_user} " for t in follow_up_triggers)

        if not is_follow_up and len(clean_user.split()) >= 4:
            return clean_user

        # Extract recent topics from history
        history_lines = [line.strip() for line in conversation_history.splitlines() if line.strip()]
        recent_context = []
        for line in reversed(history_lines[-4:]):
            if "User:" in line or "Assistant:" in line:
                recent_context.append(re.sub(r"^(User|Assistant):\s*", "", line))

        if recent_context:
            context_summary = " ".join(recent_context[:2])
            # Build combined search query
            combined_query = f"WAC {clean_user} {context_summary}"
            return re.sub(r"\s+", " ", combined_query).strip()

        return clean_user

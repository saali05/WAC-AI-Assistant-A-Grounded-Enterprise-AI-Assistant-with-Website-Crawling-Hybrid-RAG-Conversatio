import re


class ResponseFormatter:
    """
    Cleans and formats AI responses before sending them
    to the frontend.
    """

    @staticmethod
    def format(text: str) -> str:

        if not text:
            return ""

        text = ResponseFormatter.remove_markdown(text)

        text = ResponseFormatter.normalize_whitespace(text)

        text = ResponseFormatter.limit_blank_lines(text)

        text = ResponseFormatter.remove_duplicate_lines(text)

        return text.strip()

    @staticmethod
    def remove_markdown(text: str) -> str:

        # Remove headings

        text = re.sub(
            r"^#{1,6}\s*",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Remove bold

        text = text.replace("**", "")

        # Remove italic

        text = text.replace("*", "")

        # Remove horizontal rules

        text = re.sub(
            r"^-{3,}$",
            "",
            text,
            flags=re.MULTILINE,
        )

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:

        text = text.replace("\t", " ")

        text = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        return text

    @staticmethod
    def limit_blank_lines(text: str) -> str:

        return re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

    @staticmethod
    def remove_duplicate_lines(text: str) -> str:

        cleaned = []

        previous = ""

        for line in text.splitlines():

            current = line.strip()

            if current == previous:
                continue

            cleaned.append(line)

            previous = current

        return "\n".join(cleaned)
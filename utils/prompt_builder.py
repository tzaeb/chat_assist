
class PromptBuilder:
    """
    Handles the construction of prompts for the LLM based on user input, 
    uploaded files, and conversation history.
    """
    def __init__(self, context_prompt, context_search=None):
        """
        Initialize the prompt builder.
        
        Args:
            context_prompt: The base prompt for the LLM.
            context_search: An instance of ContextSearch for retrieving relevant context.
        """
        self.context_prompt = context_prompt
        self.context_search = context_search
    
    def _format_file_context(self, file_name, file_content):
        return f"""This is some additional context:
            [file name]: {file_name}
            [file content begin]
            {file_content}
            [file content end]"""

    def build_prompt(self, user_prompt, conversation_history, uploaded_file=None, file_content="", include_full_doc=False):
        """
        Build a prompt for the LLM based on the user's input, conversation history, and context.
        
        Args:
            user_prompt: The user's input.
            conversation_history: The conversation history.
            uploaded_file: The uploaded file object.
            file_content: The content of the uploaded file.
            include_full_doc: Whether to include the full document in the prompt.
            
        Returns:
            The constructed prompt.
        """
        if uploaded_file:
            # Option 1: Use full document in the prompt
            if include_full_doc:
                return (
                    f"{self.context_prompt}\n"
                    f"{self._format_file_context(uploaded_file.name, file_content)}\n"
                    f"{conversation_history}\n"
                )
            # Option 2: Retrieve top-k chunks
            elif self.context_search:
                retrieved_contexts = self.context_search.query(user_prompt, top_k=3)
                if retrieved_contexts:
                    context_text = "\n".join(
                        f"Chunk #{i+1} (Score: {res['score']:.2f}):\n{res['chunk']}"
                        for i, res in enumerate(retrieved_contexts)
                    )
                    return (
                        f"{self.context_prompt}\n"
                        f"{self._format_file_context(uploaded_file.name, context_text)}\n"
                        f"{conversation_history}\n"
                    )
        
        # No file uploaded or no relevant chunks found, just use conversation history
        return f"{self.context_prompt}\n\n{conversation_history}\n"
    
    def get_relevant_context(self, user_prompt, top_k=3):
        """
        Retrieve relevant context for the user's prompt.
        
        Args:
            user_prompt: The user's input.
            top_k: The number of chunks to retrieve.
            
        Returns:
            A list of retrieved context chunks.
        """
        if self.context_search:
            return self.context_search.query(user_prompt, top_k=top_k)
        return [] 
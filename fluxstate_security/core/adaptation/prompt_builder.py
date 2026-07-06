import json

class PromptBuilder:
    """
    Constructs deterministic, bounded semantic RAG prompts for Qwen2.5-VL.
    Ensures the context window is never bloated and bad history is ignored.
    """
    def __init__(self, max_memories=3, max_tokens_estimate=1500, min_similarity=0.85):
        self.max_memories = max_memories
        self.max_chars = max_tokens_estimate * 4 # Roughly 4 chars per token
        self.min_similarity = min_similarity

    def build_prompt(self, current_scene_description: str, rule_triggered: str, retrieved_memories: list) -> str:
        """
        Builds a structured prompt for the VLM.
        """
        # Filter and cap memories
        valid_memories = []
        for mem in retrieved_memories:
            if mem.get("similarity", 0) >= self.min_similarity:
                valid_memories.append(mem)
                
        # Enforce max memories
        valid_memories = valid_memories[:self.max_memories]

        base_instructions = (
            "Initiate Visual Threat Analysis.\n"
            f"Rule Triggered: {rule_triggered}\n"
            f"Current Scene: {current_scene_description}\n"
        )
        
        if not valid_memories:
            prompt = base_instructions + "\nAnalyze the frame based on the above information. Assess subject intent."
            return self._truncate_if_needed(prompt)

        history_block = "Historical Context (Past events that visually matched this scene):\n"
        for i, mem in enumerate(valid_memories):
            human_label = mem.get("human_label", "UNKNOWN")
            desc = mem.get("scene_description", "")
            
            # Skip malformed entries
            if not isinstance(desc, str) or not isinstance(human_label, str):
                continue
                
            history_block += f"- [{i+1}] Past Scene: {desc[:200]} | Operator Labeled it as: {human_label}\n"

        prompt = (
            base_instructions + "\n" + 
            history_block + "\n" +
            "Using the Historical Context, evaluate if the current scene is a False Positive or a True Threat. "
            "Explain your reasoning concisely."
        )

        return self._truncate_if_needed(prompt)
        
    def _truncate_if_needed(self, prompt: str) -> str:
        if len(prompt) > self.max_chars:
            return prompt[:self.max_chars] + "\n...[TRUNCATED]"
        return prompt

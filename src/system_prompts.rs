use askama::Template;
use std::fmt;

// SYSTEM PROMPTS ===================================
pub enum SystemPromptOptions{
    Default,
    ToolUse,
    RAG
}

pub fn get_system_prompt<'a>(prompt_type: &'a SystemPromptOptions) -> &'a str {
    match prompt_type {
        SystemPromptOptions::Default => {
            ""
        },
        SystemPromptOptions::ToolUse => {
            ""
        },
        SystemPromptOptions::RAG => {
            ""
        },
    }
}

// DEFAULT ================================================
#[derive(Template)]
#[template(
    ext = "txt",
    source = "\
{{ bos_token }}\
{% if let Some(sys) = system_message %}\
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ sys }}<|END_OF_TURN_TOKEN|>\
{% endif %}\
{% for message in loop_messages %}\
{% if message.role == \"user\" %}\
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message.content }}<|END_OF_TURN_TOKEN|>\
{% else if message.role == \"assistant\" %}\
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ message.content }}<|END_OF_TURN_TOKEN|>\
{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}\
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\
{% endif %}"
)]
struct DefaultPrompt<'a> {
    bos_token: String,
    system_message: Option<ChatMessage<'a>>,
    loop_messages: Vec<ChatMessage<'a>>,
    add_generation_prompt: bool,
    role: String
}

struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str 
}

impl<'a> fmt::Display for ChatMessage<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]: {}", self.role, self.content)
    }
}

//const DEFAULT_SYSTEM_PROMPT: &str = "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."
//const DEFAULT_RAG_PREAMBLE: &str = """## Task and Context You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging. ## Style Guide Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.""";

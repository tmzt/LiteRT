//! High-level conversation API with proper prompt template handling.
//!
//! Unlike the raw [`Session`](crate::Session), [`Conversation`] wraps the
//! upstream `litert_lm_conversation_*` C API which applies the model's
//! prompt template (e.g. Qwen3's `<|im_start|>user\n...<|im_end|>`) and
//! supports correct token-by-token streaming.

use std::{
    ffi::{c_char, c_void, CStr, CString},
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex},
};

use litert_lm_sys as sys;

use crate::{engine::EngineInner, input::Input, Error, Result, SamplerParams};

/// A conversation with an LLM, handling prompt formatting and multi-turn
/// context automatically.
pub struct Conversation {
    ptr: NonNull<sys::LiteRtLmConversation>,
    _engine: Arc<EngineInner>,
    /// Per-send visual token budget passed via
    /// `LiteRtLmConversationOptionalArgs`. Required for multimodal
    /// models (e.g. Gemma-4 E2B / E4B / 12B) even when only text is
    /// being sent — the C side returns `INVALID_ARGUMENT: Visual
    /// token budget must be positive` otherwise. Build via
    /// [`Self::with_visual_token_budget`]. `None` -> pass NULL
    /// optional_args (works for text-only model files).
    visual_token_budget: Option<i32>,
}

unsafe impl Send for Conversation {}

impl Conversation {
    /// Creates a new conversation from an engine with the given sampler params.
    pub(crate) fn new(engine: Arc<EngineInner>, params: SamplerParams) -> Result<Self> {
        let config = unsafe { sys::litert_lm_session_config_create() };
        if config.is_null() {
            return Err(Error::NullPointer);
        }
        let raw_params = params.to_raw();
        unsafe { sys::litert_lm_session_config_set_sampler_params(config, &raw_params) };

        // v0.13: `conversation_config_create` takes zero args; fields
        // are set via separate `_set_*` calls. We only need the
        // session_config; system message / tools / initial messages
        // default to none, which matches the old 6-arg call where
        // those slots were NULL.
        let conv_config = unsafe { sys::litert_lm_conversation_config_create() };
        if conv_config.is_null() {
            unsafe { sys::litert_lm_session_config_delete(config) };
            return Err(Error::NullPointer);
        }
        unsafe {
            sys::litert_lm_conversation_config_set_session_config(conv_config, config);
        }
        unsafe { sys::litert_lm_session_config_delete(config) };

        let conv_ptr =
            unsafe { sys::litert_lm_conversation_create(engine.ptr.as_ptr(), conv_config) };
        unsafe { sys::litert_lm_conversation_config_delete(conv_config) };

        let ptr = NonNull::new(conv_ptr).ok_or(Error::SessionCreationFailed)?;
        Ok(Self {
            ptr,
            _engine: engine,
            visual_token_budget: None,
        })
    }

    /// Set the per-send visual token budget passed in
    /// `LiteRtLmConversationOptionalArgs`. Multimodal Gemma-4 (and
    /// other vision-capable) `.litertlm` files require this to be
    /// positive even for text-only sends — otherwise the C side
    /// returns `INVALID_ARGUMENT: Visual token budget must be positive`.
    /// 512 is a reasonable default for text-only chat; higher values
    /// reserve more KV cache budget for image tokens.
    pub fn with_visual_token_budget(mut self, budget: i32) -> Self {
        self.visual_token_budget = Some(budget);
        self
    }

    /// Sends a message and streams the response token-by-token.
    ///
    /// `on_token` receives each text chunk as it's generated. The model's
    /// prompt template is applied automatically. Blocks until generation
    /// completes.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use litertlm::{Engine, EngineSettings, SamplerParams};
    /// # fn demo(engine: &Engine) -> litertlm::Result<()> {
    /// let mut conv = engine.create_conversation(SamplerParams::default().top_p(0.95))?;
    /// conv.send_message_stream("Tell me a story", |chunk| {
    ///     print!("{chunk}");
    /// })?;
    /// # Ok(()) }
    /// ```
    pub fn send_message_stream(&mut self, prompt: &str, on_token: impl FnMut(&str)) -> Result<()> {
        let message_json = format!(
            r#"{{"role":"user","content":[{{"type":"text","text":{}}}]}}"#,
            serde_json_escape(prompt)
        );
        self.send_raw_stream(&message_json, on_token)
    }

    /// Sends a pre-formatted JSON message and streams the response.
    ///
    /// Use this when you need full control over the message JSON, e.g. for
    /// multimodal inputs with image file paths.
    pub fn send_raw_stream(
        &mut self,
        message_json: &str,
        mut on_token: impl FnMut(&str),
    ) -> Result<()> {
        let msg_cstr = CString::new(message_json).map_err(|_| Error::NullPointer)?;

        struct State<'a> {
            cb: &'a mut dyn FnMut(&str),
            error: Option<String>,
            done: &'a Mutex<bool>,
            cond: &'a Condvar,
        }

        unsafe extern "C" fn trampoline(
            data: *mut c_void,
            chunk: *const c_char,
            is_final: bool,
            error_msg: *const c_char,
        ) {
            let state = &mut *(data as *mut State);
            if !error_msg.is_null() {
                state.error = Some(CStr::from_ptr(error_msg).to_string_lossy().into_owned());
                *state.done.lock().unwrap() = true;
                state.cond.notify_one();
                return;
            }
            if !chunk.is_null() {
                let raw = CStr::from_ptr(chunk).to_string_lossy();
                let text = extract_text_from_json(&raw).unwrap_or_else(|| raw.to_string());
                if !text.is_empty() {
                    (state.cb)(&text);
                }
            }
            if is_final {
                *state.done.lock().unwrap() = true;
                state.cond.notify_one();
            }
        }

        let done = Mutex::new(false);
        let cond = Condvar::new();
        let mut state = State {
            cb: &mut on_token,
            error: None,
            done: &done,
            cond: &cond,
        };

        // Build OptionalArgs iff a visual_token_budget was set on this
        // Conversation. Multimodal Gemma-4 models require it (see
        // [`Self::with_visual_token_budget`]). The args object is
        // stack-owned by the C wrapper and we delete after the call.
        let optional_args = if let Some(budget) = self.visual_token_budget {
            let oa = unsafe { sys::litert_lm_conversation_optional_args_create() };
            if !oa.is_null() {
                unsafe {
                    sys::litert_lm_conversation_optional_args_set_visual_token_budget(oa, budget);
                }
            }
            oa
        } else {
            std::ptr::null_mut()
        };

        let ret = unsafe {
            sys::litert_lm_conversation_send_message_stream(
                self.ptr.as_ptr(),
                msg_cstr.as_ptr(),
                std::ptr::null(),
                optional_args,
                Some(trampoline),
                &mut state as *mut State as *mut c_void,
            )
        };

        if !optional_args.is_null() {
            unsafe { sys::litert_lm_conversation_optional_args_delete(optional_args) };
        }

        if ret != 0 {
            return Err(Error::GenerationFailed(format!(
                "conversation stream returned {ret}"
            )));
        }

        let guard = done.lock().unwrap();
        let _guard = cond.wait_while(guard, |d| !*d).unwrap();

        if let Some(err) = state.error {
            return Err(Error::GenerationFailed(err));
        }
        Ok(())
    }

    /// Sends a message and returns the full response (blocking).
    pub fn send_message(&mut self, prompt: &str) -> Result<String> {
        let mut response = String::new();
        self.send_message_stream(prompt, |chunk| {
            response.push_str(chunk);
        })?;
        Ok(response)
    }

    /// Sends multimodal inputs (text + images + audio) with streaming.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use litertlm::{Engine, EngineSettings, SamplerParams, Input};
    /// # fn demo(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut conv = engine.create_conversation(SamplerParams::default().top_p(0.95))?;
    /// let image = std::fs::read("photo.jpg")?;
    /// conv.send_inputs_stream(
    ///     &[Input::image(&image), Input::text("What's in this image?")],
    ///     |chunk| print!("{chunk}"),
    /// )?;
    /// # Ok(()) }
    /// ```
    pub fn send_inputs_stream(
        &mut self,
        inputs: &[Input<'_>],
        on_token: impl FnMut(&str),
    ) -> Result<()> {
        let content_json = crate::input::inputs_to_content_json(inputs);
        let message_json = format!(r#"{{"role":"user","content":{content_json}}}"#);
        self.send_raw_stream(&message_json, on_token)
    }

    /// Sends multimodal inputs and returns the full response (blocking).
    pub fn send_inputs(&mut self, inputs: &[Input<'_>]) -> Result<String> {
        let mut response = String::new();
        self.send_inputs_stream(inputs, |chunk| {
            response.push_str(chunk);
        })?;
        Ok(response)
    }
}

impl Drop for Conversation {
    fn drop(&mut self) {
        unsafe { sys::litert_lm_conversation_delete(self.ptr.as_ptr()) }
    }
}

/// Minimal JSON string escaping for the message payload.
pub(crate) fn serde_json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Try to extract text from a conversation JSON chunk.
/// Format: `{"content":[{"type":"text","text":"..."}]}` or just `{"text":"..."}`
fn extract_text_from_json(raw: &str) -> Option<String> {
    // Quick check: if it doesn't look like JSON, return None
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    // Naive extraction: find "text":" and extract the value
    let marker = r#""text":""#;
    let start = trimmed.find(marker)? + marker.len();
    let rest = &trimmed[start..];
    // Find the closing quote (handling escaped quotes)
    let mut end = 0;
    let mut escape = false;
    for c in rest.chars() {
        if escape {
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == '"' {
            break;
        }
        end += c.len_utf8();
    }
    Some(rest[..end].replace("\\n", "\n").replace("\\\"", "\""))
}

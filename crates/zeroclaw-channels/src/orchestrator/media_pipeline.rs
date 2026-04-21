//! Automatic media understanding pipeline for inbound channel messages.
//!
//! Pre-processes media attachments (audio, images, video) before the agent sees
//! the message, enriching the text with human-readable annotations:
//!
//! - **Audio**: transcribed via the existing [`super::transcription`] infrastructure,
//!   prepended as `[Audio transcription: ...]`.
//! - **Images**: when a vision-capable provider is active, described as `[Image: <description>]`.
//!   Falls back to `[Image: attached]` when vision is unavailable.
//! - **Video**: summarised as `[Video summary: ...]` when an API is available,
//!   otherwise `[Video: attached]`.
//!
//! The pipeline is **opt-in** via `[media_pipeline] enabled = true` in config.

use zeroclaw_config::schema::{MediaPipelineConfig, TranscriptionConfig};

// Re-export media types from zeroclaw-types for backwards compatibility.
pub use zeroclaw_api::media::{MediaAttachment, MediaKind};

const IMAGE_MARKER_PREFIX: &str = "[IMAGE:";

/// The media understanding pipeline.
///
/// Consumes a message's text and attachments, returning enriched text with
/// media annotations prepended.
pub struct MediaPipeline<'a> {
    config: &'a MediaPipelineConfig,
    transcription_config: &'a TranscriptionConfig,
    vision_available: bool,
}

impl<'a> MediaPipeline<'a> {
    /// Create a new pipeline. `vision_available` indicates whether the current
    /// provider supports vision (image description).
    pub fn new(
        config: &'a MediaPipelineConfig,
        transcription_config: &'a TranscriptionConfig,
        vision_available: bool,
    ) -> Self {
        Self {
            config,
            transcription_config,
            vision_available,
        }
    }

    /// Process a message's attachments and return enriched text.
    ///
    /// If the pipeline is disabled via config, returns `original_text` unchanged.
    ///
    /// When vision is unavailable and image attachments are present, raw
    /// `[IMAGE:]` markers are stripped from the text to prevent text-only
    /// providers from throwing `provider_capability_error`.
    pub async fn process(&self, original_text: &str, attachments: &[MediaAttachment]) -> String {
        if !self.config.enabled || attachments.is_empty() {
            return original_text.to_string();
        }

        let mut annotations = Vec::new();
        let mut has_images = false;

        for attachment in attachments {
            match attachment.kind() {
                MediaKind::Audio if self.config.transcribe_audio => {
                    let annotation = self.process_audio(attachment).await;
                    annotations.push(annotation);
                }
                MediaKind::Image if self.config.describe_images => {
                    let annotation = self.process_image(attachment);
                    annotations.push(annotation);
                    has_images = true;
                }
                MediaKind::Video if self.config.summarize_video => {
                    let annotation = self.process_video(attachment);
                    annotations.push(annotation);
                }
                _ => {}
            }
        }

        if annotations.is_empty() {
            return original_text.to_string();
        }

        // When vision is unavailable and we processed images, strip raw [IMAGE:]
        // markers from the text so text-only providers don't error on them.
        let text_for_output = if !self.vision_available && has_images {
            strip_image_markers(original_text)
        } else {
            original_text.to_string()
        };

        let mut enriched = String::with_capacity(
            annotations.iter().map(|a| a.len() + 1).sum::<usize>() + text_for_output.len() + 2,
        );

        for annotation in &annotations {
            enriched.push_str(annotation);
            enriched.push('\n');
        }

        if !text_for_output.is_empty() {
            enriched.push('\n');
            enriched.push_str(&text_for_output);
        }

        enriched.trim().to_string()
    }

    /// Transcribe an audio attachment using the existing transcription infra.
    async fn process_audio(&self, attachment: &MediaAttachment) -> String {
        if !self.transcription_config.enabled {
            return "[Audio: attached]".to_string();
        }

        match super::transcription::transcribe_audio(
            attachment.data.clone(),
            &attachment.file_name,
            self.transcription_config,
        )
        .await
        {
            Ok(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    "[Audio transcription: (empty)]".to_string()
                } else {
                    format!("[Audio transcription: {trimmed}]")
                }
            }
            Err(err) => {
                tracing::warn!(
                    file = %attachment.file_name,
                    error = %err,
                    "Media pipeline: audio transcription failed"
                );
                "[Audio: transcription failed]".to_string()
            }
        }
    }

    /// Describe an image attachment.
    ///
    /// When vision is available, the image will be passed through to the
    /// provider as an `[IMAGE:]` marker and described by the model in the
    /// normal flow. Here we only add a placeholder annotation so the agent
    /// knows an image is present.
    fn process_image(&self, attachment: &MediaAttachment) -> String {
        if self.vision_available {
            format!(
                "[Image: {} attached, will be processed by vision model]",
                attachment.file_name
            )
        } else {
            format!("[Image: {} attached]", attachment.file_name)
        }
    }

    /// Summarize a video attachment.
    ///
    /// Video analysis requires external APIs not currently integrated.
    /// For now we add a placeholder annotation.
    fn process_video(&self, attachment: &MediaAttachment) -> String {
        format!("[Video: {} attached]", attachment.file_name)
    }
}

/// Strip all `[IMAGE:...]` markers from text.
///
/// Used when vision is unavailable so text-only providers don't try to
/// process image references they can't handle.
fn strip_image_markers(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut cursor = 0usize;

    while let Some(rel_start) = text[cursor..].find(IMAGE_MARKER_PREFIX) {
        let start = cursor + rel_start;
        result.push_str(&text[cursor..start]);

        let marker_start = start + IMAGE_MARKER_PREFIX.len();
        if let Some(rel_end) = text[marker_start..].find(']') {
            let end = marker_start + rel_end;
            cursor = end + 1;
        } else {
            // No closing bracket found, consume rest of text as-is.
            result.push_str(&text[start..]);
            return result;
        }
    }

    result.push_str(&text[cursor..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline_config(enabled: bool) -> MediaPipelineConfig {
        MediaPipelineConfig {
            enabled,
            transcribe_audio: true,
            describe_images: true,
            summarize_video: true,
        }
    }

    fn sample_audio() -> MediaAttachment {
        MediaAttachment {
            file_name: "voice.ogg".to_string(),
            data: vec![0u8; 100],
            mime_type: Some("audio/ogg".to_string()),
        }
    }

    fn sample_image() -> MediaAttachment {
        MediaAttachment {
            file_name: "photo.jpg".to_string(),
            data: vec![0u8; 50],
            mime_type: Some("image/jpeg".to_string()),
        }
    }

    fn sample_video() -> MediaAttachment {
        MediaAttachment {
            file_name: "clip.mp4".to_string(),
            data: vec![0u8; 200],
            mime_type: Some("video/mp4".to_string()),
        }
    }

    #[test]
    fn media_kind_from_mime() {
        let audio = MediaAttachment {
            file_name: "file".to_string(),
            data: vec![],
            mime_type: Some("audio/ogg".to_string()),
        };
        assert_eq!(audio.kind(), MediaKind::Audio);

        let image = MediaAttachment {
            file_name: "file".to_string(),
            data: vec![],
            mime_type: Some("image/png".to_string()),
        };
        assert_eq!(image.kind(), MediaKind::Image);

        let video = MediaAttachment {
            file_name: "file".to_string(),
            data: vec![],
            mime_type: Some("video/mp4".to_string()),
        };
        assert_eq!(video.kind(), MediaKind::Video);
    }

    #[test]
    fn media_kind_from_extension() {
        let audio = MediaAttachment {
            file_name: "voice.ogg".to_string(),
            data: vec![],
            mime_type: None,
        };
        assert_eq!(audio.kind(), MediaKind::Audio);

        let image = MediaAttachment {
            file_name: "photo.png".to_string(),
            data: vec![],
            mime_type: None,
        };
        assert_eq!(image.kind(), MediaKind::Image);

        let video = MediaAttachment {
            file_name: "clip.mp4".to_string(),
            data: vec![],
            mime_type: None,
        };
        assert_eq!(video.kind(), MediaKind::Video);

        let unknown = MediaAttachment {
            file_name: "data.bin".to_string(),
            data: vec![],
            mime_type: None,
        };
        assert_eq!(unknown.kind(), MediaKind::Unknown);
    }

    #[tokio::test]
    async fn disabled_pipeline_returns_original_text() {
        let config = default_pipeline_config(false);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let result = pipeline.process("hello", &[sample_audio()]).await;
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn empty_attachments_returns_original_text() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let result = pipeline.process("hello", &[]).await;
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn image_annotation_with_vision() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, true);

        let result = pipeline.process("check this", &[sample_image()]).await;
        assert!(
            result.contains("[Image: photo.jpg attached, will be processed by vision model]"),
            "expected vision annotation, got: {result}"
        );
        assert!(result.contains("check this"));
    }

    #[tokio::test]
    async fn image_annotation_without_vision() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let result = pipeline.process("check this", &[sample_image()]).await;
        assert!(
            result.contains("[Image: photo.jpg attached]"),
            "expected basic image annotation, got: {result}"
        );
    }

    #[tokio::test]
    async fn video_annotation() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let result = pipeline.process("watch", &[sample_video()]).await;
        assert!(
            result.contains("[Video: clip.mp4 attached]"),
            "expected video annotation, got: {result}"
        );
    }

    #[tokio::test]
    async fn audio_without_transcription_enabled() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig {
            enabled: false,
            ..Default::default()
        };
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let result = pipeline.process("", &[sample_audio()]).await;
        assert_eq!(result, "[Audio: attached]");
    }

    #[tokio::test]
    async fn multiple_attachments_produce_multiple_annotations() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig {
            enabled: false,
            ..Default::default()
        };
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let attachments = vec![sample_audio(), sample_image(), sample_video()];
        let result = pipeline.process("context", &attachments).await;

        assert!(
            result.contains("[Audio: attached]"),
            "missing audio annotation"
        );
        assert!(
            result.contains("[Image: photo.jpg attached]"),
            "missing image annotation"
        );
        assert!(
            result.contains("[Video: clip.mp4 attached]"),
            "missing video annotation"
        );
        assert!(result.contains("context"), "missing original text");
    }

    #[tokio::test]
    async fn disabled_sub_features_skip_processing() {
        let config = MediaPipelineConfig {
            enabled: true,
            transcribe_audio: false,
            describe_images: false,
            summarize_video: false,
        };
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let attachments = vec![sample_audio(), sample_image(), sample_video()];
        let result = pipeline.process("hello", &attachments).await;
        assert_eq!(result, "hello");
    }

    // ── [IMAGE:] marker stripping when vision unavailable ──────────────────────

    /// When vision is unavailable, raw `[IMAGE:/path]` markers in the text must
    /// be stripped so text-only providers don't throw provider_capability_error.
    /// The placeholder annotation is prepended instead.
    #[tokio::test]
    async fn image_marker_stripped_when_vision_unavailable() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        // Simulate text that already has [IMAGE:/path] marker (as Telegram would insert).
        let result = pipeline
            .process(
                "Look at this [IMAGE:/tmp/workspace/photo_99_1.jpg] and tell me what you see",
                &[sample_image()],
            )
            .await;

        // Raw [IMAGE:] marker must NOT appear in output.
        assert!(
            !result.contains("[IMAGE:"),
            "raw [IMAGE:] marker should be stripped, got: {result}"
        );
        // Placeholder annotation should be present.
        assert!(
            result.contains("[Image: photo.jpg attached]"),
            "expected placeholder annotation, got: {result}"
        );
        // Original text content (minus the marker) should be preserved.
        assert!(
            result.contains("Look at this"),
            "original text context should be preserved, got: {result}"
        );
        assert!(
            result.contains("tell me what you see"),
            "original text trailing content should be preserved, got: {result}"
        );
    }

    /// When vision is available, raw `[IMAGE:/path]` markers must be preserved
    /// so the vision pipeline can actually process them.
    #[tokio::test]
    async fn image_marker_preserved_when_vision_available() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, true);

        let result = pipeline
            .process(
                "Look at [IMAGE:/tmp/workspace/photo_99_1.jpg]",
                &[sample_image()],
            )
            .await;

        // Raw marker must be preserved when vision is available.
        assert!(
            result.contains("[IMAGE:/tmp/workspace/photo_99_1.jpg]"),
            "raw [IMAGE:] marker should be preserved when vision available, got: {result}"
        );
        // Vision-specific annotation should be present.
        assert!(
            result.contains("will be processed by vision model"),
            "expected vision annotation, got: {result}"
        );
    }

    /// Multiple `[IMAGE:]` markers in text are all stripped when vision unavailable.
    #[tokio::test]
    async fn multiple_image_markers_stripped_when_vision_unavailable() {
        let config = default_pipeline_config(true);
        let tc = TranscriptionConfig::default();
        let pipeline = MediaPipeline::new(&config, &tc, false);

        let attachments = vec![sample_image(), sample_image()];
        let result = pipeline
            .process(
                "First [IMAGE:/tmp/a.png] then [IMAGE:/tmp/b.jpg] done",
                &attachments,
            )
            .await;

        assert!(
            !result.contains("[IMAGE:"),
            "no [IMAGE:] markers should remain, got: {result}"
        );
        // Both images should have annotations.
        assert!(
            result.contains("[Image: photo.jpg attached]"),
            "expected image annotations, got: {result}"
        );
    }
}
